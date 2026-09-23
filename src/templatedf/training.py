"""Budgeted 1A training on supplied cached/tensor datasets; no online ESM."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import atomic_save, capture_rng, load_checkpoint, restore_rng, seed_all
from .data import AA_ORDER, PAD_LABEL
from .losses import MetricAccumulator, reconstruction_loss
from .model import ProteinAutoencoder


@dataclass
class TrainingSettings:
    stage: str='1A'
    precision: str='fp32'
    device: str='cpu'
    batch_size: int=8
    lambda_emb: float=0.0
    max_steps: int | None=None
    learning_rate: float=1e-4
    weight_decay: float=0.01
    grad_clip: float=1.0
    seed: int=42
    eval_every: int=100
    save_every: int=100
    scheduler_gamma: float=1.0

    def __post_init__(self):
        if self.stage!='1A':
            raise NotImplementedError(f"Unsupported training stage: {self.stage}; only 1A is implemented")
        if self.precision not in ('fp32','bf16'):
            raise ValueError("precision must be fp32 or bf16")
        for name in ('batch_size','max_steps','eval_every','save_every'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:
                raise ValueError(f"{name} must be an explicit positive integer")
        if type(self.seed) is not int or not 0<=self.seed<2**32:
            raise ValueError("seed must be an integer in [0,2**32)")
        for name in ('lambda_emb','weight_decay','learning_rate','grad_clip','scheduler_gamma'):
            value=getattr(self,name)
            if not math.isfinite(value) or value<0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.learning_rate==0 or self.grad_clip==0 or not 0<self.scheduler_gamma<=1:
            raise ValueError("learning_rate/grad_clip must be positive; scheduler_gamma in (0,1]")


def model_settings(config):
    values=deepcopy(config.get('model',{}))
    if values.pop('condition',None) is not None:
        raise NotImplementedError("Training condition must be None")
    if values.pop('encoder_ffn_ratio',2)!=2:
        raise ValueError("Encoder FFN ratio is fixed at 2")
    return values


def collate_training_samples(items):
    """Dynamic-width collation for cached 1280-D or explicitly synthetic test data."""
    if not items:
        raise ValueError("Empty training batch")
    dim=items[0]['embeddings'].shape[-1]
    lengths=torch.tensor([len(item['labels']) for item in items],dtype=torch.long)
    if not (lengths>0).all():
        raise ValueError("Empty peptide sample")
    maximum=int(lengths.max())
    dtype=items[0]["embeddings"].dtype
    if dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise ValueError("Expected floating cached embeddings")
    embeddings=torch.zeros(len(items),maximum,dim,dtype=dtype)
    labels=torch.full((len(items),maximum),PAD_LABEL,dtype=torch.long)
    for index,item in enumerate(items):
        length=int(lengths[index])
        if item['embeddings'].shape!=(length,dim) or item['labels'].dtype not in (torch.long,torch.int8):
            raise ValueError("Invalid cached tensor shapes or labels")
        if item['embeddings'].dtype != dtype:
            raise ValueError('Mixed embedding dtypes in one batch')
        embeddings[index,:length]=item['embeddings'].detach().cpu()
        labels[index,:length]=item['labels'].cpu()
    return {'embeddings':embeddings,'labels':labels,'lengths':lengths,
            'input_mask':torch.arange(maximum)[None,:]<lengths[:,None]}


class StatefulBatchSampler:
    """Single-process shuffled stream with saved permutation and exact cursor."""
    def __init__(self, size, batch_size, seed):
        if size<1:
            raise ValueError("Training dataset must be nonempty")
        self.size,self.batch_size=size,batch_size
        self.generator=torch.Generator().manual_seed(seed)
        self.order=torch.randperm(size,generator=self.generator)
        self.cursor=0
        self.epoch=0

    def next_indices(self):
        if self.cursor==self.size:
            self.order=torch.randperm(self.size,generator=self.generator)
            self.cursor=0
            self.epoch+=1
        end=min(self.cursor+self.batch_size,self.size)
        result=self.order[self.cursor:end].tolist()
        self.cursor=end
        self.last_indices=result
        return result

    def __iter__(self):
        # One iterator is one (possibly resumed partial) epoch; no worker prefetch.
        if self.cursor == self.size:
            self.order=torch.randperm(self.size,generator=self.generator)
            self.cursor=0
            self.epoch+=1
        while self.cursor < self.size:
            yield self.next_indices()

    def state_dict(self):
        return {'size':self.size,'batch_size':self.batch_size,'order':self.order.clone(),
                'cursor':self.cursor,'epoch':self.epoch,'generator':self.generator.get_state()}

    def load_state_dict(self,state):
        if state['size']!=self.size or state['batch_size']!=self.batch_size:
            raise ValueError("Resume sampler dataset size/batch size mismatch")
        if (state['order'].shape!=(self.size,) or state['order'].dtype!=torch.long
                or not torch.equal(state['order'].sort().values,torch.arange(self.size))
                or not 0<=state['cursor']<=self.size):
            raise ValueError("Invalid saved sampler permutation/cursor")
        self.order=state['order'].clone()
        self.cursor=state['cursor']
        self.epoch=state['epoch']
        self.generator.set_state(state['generator'].cpu())


def make_data_loader(dataset, *, device, batch_size=8, batch_sampler=None, seed=42, options=None):
    settings={'num_workers':0,'persistent_workers':False,'pin_memory':True}
    settings.update(options or {})
    if set(settings)!={'num_workers','persistent_workers','pin_memory'}:
        raise ValueError('Unsupported loader option; do not set prefetch_factor with num_workers=0')
    if settings['num_workers'] != 0 or settings['persistent_workers'] is not False:
        raise ValueError('RAM cache loader supports num_workers=0, persistent_workers=False')
    cuda=torch.device(device).type=='cuda'
    if not isinstance(settings['pin_memory'],bool) or (cuda and not settings['pin_memory']):
        raise ValueError('CUDA RAM-cache batches require pin_memory=True')
    kwargs=dict(num_workers=0,persistent_workers=False,pin_memory=cuda and settings['pin_memory'],
                collate_fn=collate_training_samples,generator=torch.Generator().manual_seed(seed))
    # Do not pass prefetch_factor at all. Private generator avoids changing model RNG.
    if batch_sampler is not None:
        kwargs['batch_sampler']=batch_sampler
    else:
        kwargs.update(batch_size=batch_size,shuffle=False)
    return DataLoader(dataset,**kwargs)


def batch_to_device(batch, device):
    device=torch.device(device)
    moved={key:value.to(device,non_blocking=device.type=='cuda') for key,value in batch.items()}
    # Cache and host batch stay FP16. Upcast only this batch after GPU transfer,
    # matching the existing FP32 input + BF16 autocast model/loss strategy.
    moved['embeddings']=moved['embeddings'].float()
    return moved


class TrainingEngine:
    def __init__(self,config,train_dataset,validation_dataset,*,data_provenance,
                 init_checkpoint=None,resume=None):
        if init_checkpoint is not None and resume is not None:
            raise ValueError("init_checkpoint and resume are mutually exclusive")
        self.config=deepcopy(config)
        self.settings=TrainingSettings(**config.get('training',{}))
        self.model_config=model_settings(config)
        self.device=torch.device(self.settings.device)
        if self.device.type not in ('cpu','cuda'):
            raise ValueError("Only CPU and CUDA are supported")
        if self.device.type=='cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            self.device=torch.device('cuda',self.device.index if self.device.index is not None else torch.cuda.current_device())
        self.precision_runtime={'requested':self.settings.precision,'device':str(self.device),
                                'autocast_dtype':'torch.bfloat16' if self.settings.precision=='bf16' else None,
                                'grad_scaler':None}
        if self.settings.precision=='bf16':
            if self.device.type!='cuda':
                raise RuntimeError("BF16 training requires a CUDA GPU with native BF16 support")
            with torch.cuda.device(self.device):
                supported=torch.cuda.is_bf16_supported(including_emulation=False)
            if not supported:
                raise RuntimeError("Native BF16 is not supported; no silent FP32 fallback")
            self.precision_runtime['native_bf16_supported']=True
        if not len(train_dataset) or not len(validation_dataset):
            raise ValueError("Separate nonempty train and validation datasets are required")
        if train_dataset is validation_dataset:
            raise ValueError("Train and validation datasets must be separate")
        if not data_provenance.get('esm_source') or data_provenance.get('data_kind') not in ('synthetic','real_cached'):
            raise ValueError("Explicit ESM source and data_kind provenance are required")
        if not data_provenance.get('train') or not data_provenance.get('validation'):
            raise ValueError("Both dataset identities must be recorded")
        self.data_provenance=deepcopy(data_provenance)
        self.train_dataset,self.validation_dataset=train_dataset,validation_dataset
        seed_all(self.settings.seed)
        self.model=ProteinAutoencoder(**self.model_config).to(self.device)
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=self.settings.learning_rate,
                                         weight_decay=self.settings.weight_decay)
        self.scheduler=(torch.optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=self.settings.scheduler_gamma)
                        if self.settings.scheduler_gamma!=1 else None)
        self.sampler=StatefulBatchSampler(len(train_dataset),self.settings.batch_size,self.settings.seed)
        self.step=0
        self.is_resume=resume is not None
        if resume is not None:
            self.restore(resume)
        elif init_checkpoint is not None:
            state=load_checkpoint(init_checkpoint)
            self._validate_identity(state)
            self.model.load_state_dict(state['model'],strict=True)
        loader_options=config.get('dataloader',{})
        self.train_loader=make_data_loader(train_dataset,device=self.device,batch_sampler=self.sampler,
                                           seed=self.settings.seed,options=loader_options)
        self.validation_loader=make_data_loader(validation_dataset,device=self.device,batch_size=self.settings.batch_size,
                                                seed=self.settings.seed+1,options=loader_options)
        self._train_iterator=None

    def _autocast(self):
        return torch.autocast('cuda',dtype=torch.bfloat16) if self.settings.precision=='bf16' else nullcontext()

    def _batch(self,dataset,indices):
        batch=collate_training_samples([dataset[i] for i in indices])
        if self.device.type=='cuda':
            batch={key:value.pin_memory() for key,value in batch.items()}
        return batch_to_device(batch,self.device)

    def _forward_loss(self,batch):
        # Labels are passed only to the loss, never to the autoencoder.
        output=self.model(batch['embeddings'],batch['input_mask'],output_lengths=batch['lengths'],condition=None)
        if not torch.equal(output['output_mask'],batch['input_mask']):
            raise RuntimeError("1A output mask must equal input mask")
        expected_dtype=torch.bfloat16 if self.settings.precision=='bf16' else torch.float32
        if output['logits'].dtype!=expected_dtype or output['reconstructed_embeddings'].dtype!=expected_dtype:
            raise RuntimeError("Requested training precision did not take effect")
        self.precision_runtime.update(logits_dtype=str(output['logits'].dtype),
                                      embeddings_dtype=str(output['reconstructed_embeddings'].dtype),
                                      autocast_enabled=torch.is_autocast_enabled('cuda'))
        return reconstruction_loss(output['logits'],output['reconstructed_embeddings'],batch['embeddings'],
                                   batch['labels'],output['output_mask'],lambda_emb=self.settings.lambda_emb)

    def train_step(self):
        if self.step>=self.settings.max_steps:
            raise RuntimeError("max_steps budget exhausted")
        self.model.train()
        if self._train_iterator is None:
            self._train_iterator=iter(self.train_loader)
        try:
            host_batch=next(self._train_iterator)
        except StopIteration:
            self._train_iterator=iter(self.train_loader)
            host_batch=next(self._train_iterator)
        indices=list(self.sampler.last_indices)
        batch=batch_to_device(host_batch,self.device)
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            metrics=self._forward_loss(batch)
        metrics['loss'].backward()
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.settings.grad_clip,error_if_nonfinite=True)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.step+=1
        accumulator=MetricAccumulator()
        accumulator.update(metrics)
        return {**accumulator.compute(self.settings.lambda_emb),'gradient_norm_before_clip':float(norm),
                'learning_rate':self.optimizer.param_groups[0]['lr'],'sample_indices':indices,
                'precision':deepcopy(self.precision_runtime)}

    @torch.no_grad()
    def evaluate(self):
        previous_mode=self.model.training
        rng=capture_rng(self.device)
        self.model.eval()
        accumulator=MetricAccumulator()
        try:
            for host_batch in self.validation_loader:
                with self._autocast():
                    metrics=self._forward_loss(batch_to_device(host_batch,self.device))
                accumulator.update(metrics)
            return accumulator.compute(self.settings.lambda_emb)
        finally:
            self.model.train(previous_mode)
            restore_rng(rng,self.device)

    def _validate_identity(self,state):
        if state['stage']!='1A' or state['aa_order']!=AA_ORDER:
            raise ValueError("Checkpoint stage or AA order mismatch")
        if state['model_config']!=self.model_config:
            raise ValueError("Checkpoint model configuration mismatch")
        if state['esm_source']!=self.data_provenance['esm_source']:
            raise ValueError("Checkpoint ESM source mismatch")

    def restore(self,path):
        state=load_checkpoint(path)
        self._validate_identity(state)
        if state['data_provenance']!=self.data_provenance:
            raise ValueError("Resume dataset provenance mismatch")
        old=state['training_settings'].copy()
        new=asdict(self.settings)
        for key in ('max_steps','eval_every','save_every'):
            old.pop(key);new.pop(key)
        if old!=new:
            raise ValueError("Resume training settings mismatch (only budget/log intervals may change)")
        if type(state['step']) is not int or not 0<=state['step']<=self.settings.max_steps:
            raise ValueError("Resume step exceeds max_steps or is invalid")
        if state['scaler'] is not None:
            raise ValueError("FP32/BF16 trainer does not use a GradScaler")
        if (state['scheduler'] is None)!=(self.scheduler is None):
            raise ValueError("Resume scheduler mismatch")
        self.model.load_state_dict(state['model'],strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler'])
        self.sampler.load_state_dict(state['sampler'])
        self.step=state['step']
        self.precision_runtime=state['precision_runtime']
        restore_rng(state['rng'],self.device)
        self.is_resume=True
        self._train_iterator=None

    def save(self,path):
        atomic_save({'format_version':1,'stage':'1A','config':deepcopy(self.config),
                     'model_config':self.model_config,'training_settings':asdict(self.settings),
                     'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),
                     'scheduler':self.scheduler.state_dict() if self.scheduler is not None else None,
                     'scaler':None,'step':self.step,'rng':capture_rng(self.device),
                     'sampler':self.sampler.state_dict(),'aa_order':AA_ORDER,
                     'esm_source':self.data_provenance['esm_source'],'data_provenance':self.data_provenance,
                     'precision_runtime':self.precision_runtime},path)

    def fit(self,run_dir):
        run_dir=Path(run_dir)
        if not self.is_resume and any((run_dir/name).exists() for name in ('metrics.jsonl','last.pt','summary.json')):
            raise FileExistsError("Existing run artifacts: choose a new output directory or use resume")
        run_dir.mkdir(parents=True,exist_ok=True)
        def log(event,values):
            row={'event':event,'step':self.step,'data_kind':self.data_provenance['data_kind'],**values}
            with (run_dir/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
        log('configuration',{'settings':asdict(self.settings),'precision':self.precision_runtime,
                             'dataloader':{'num_workers':0,'persistent_workers':False,'pin_memory':self.train_loader.pin_memory},
                             'resume':self.is_resume,'esm_alignment_supervised':self.settings.lambda_emb>0})
        initial=self.evaluate()
        log('validation',initial)
        last_validation=initial
        while self.step<self.settings.max_steps:
            log('train',self.train_step())
            if self.step%self.settings.eval_every==0 or self.step==self.settings.max_steps:
                last_validation=self.evaluate()
                log('validation',last_validation)
            if self.step%self.settings.save_every==0:
                self.save(run_dir/'last.pt')
        self.save(run_dir/'last.pt')
        summary={'step':self.step,'max_steps':self.settings.max_steps,'initial_validation':initial,
                 'final_validation':last_validation,'precision':self.precision_runtime,
                 'data_kind':self.data_provenance['data_kind'],
                 'esm_alignment_supervised':self.settings.lambda_emb>0,
                 'checkpoint':str((run_dir/'last.pt').resolve())}
        (run_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        return summary
