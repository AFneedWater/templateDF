"""Offline stage 06 checks: template dependence, RNG, strict input and shards."""
from copy import deepcopy
import importlib.abc
import json
import subprocess
import sys

import pytest
import torch
from torch import nn

from templatedf.data import AA_ORDER, PeptideRecord, encode_labels, file_sha256
from templatedf.inference import TemplateGenerator, sample_logits
from templatedf.inference_io import (export_raw_latents, inspect_fasta, iter_latent_records,
                                    selected_records, write_candidates)
from templatedf.model import ProteinAutoencoder


class MockESM:
    def __init__(self):
        self.model=nn.Linear(32,32).eval().requires_grad_(False)
        self.calls=[]

    def extract(self, records):
        assert not torch.is_grad_enabled()
        self.model.eval().requires_grad_(False)
        self.calls.append([r.sequence for r in records])
        features=[]
        for r in records:
            value=torch.nn.functional.one_hot(encode_labels(r.sequence),num_classes=32).float()
            value[:,20]=torch.arange(r.length)/150
            features.append(value)
        return features


@pytest.fixture
def runtime():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(61)
        cfg=dict(dim=32,num_latents=50,num_heads=4,encoder_blocks=1,decoder_blocks=1,dropout=.2)
        model=ProteinAutoencoder(**cfg)
        return TemplateGenerator(model,MockESM(),provenance={
            'aa_order':AA_ORDER,'checkpoint':{'path':'mock:unit','sha256':'a'*64,'step':0,'stage':'1A'},
            'esm_source':{'source':'mock:onehot','storage_dtype':'float32'},'model_config':cfg,'config':{'model':cfg}})


def templates():
    return [PeptideRecord('same-id','A'*10),PeptideRecord('same-id','C'*23),PeptideRecord('third','ACDEFGHIKL'*15)]


def test_actual_encoder_latent_is_passed_to_decoder_and_depends_on_template(runtime):
    encoded=[];decoded=[]
    def enc_hook(module,args,output):
        assert not module.training and not torch.is_grad_enabled()
        encoded.append(output.detach().clone())
    def dec_hook(module,args):
        assert not module.training and not torch.is_grad_enabled()
        assert len(args)==2
        decoded.append(args[0].detach().clone())
    h1=runtime.model.encoder.register_forward_hook(enc_hook)
    h2=runtime.model.decoder.register_forward_pre_hook(dec_hook)
    records=[PeptideRecord('a','A'*10),PeptideRecord('c','C'*10)]
    runtime.model.train()  # API must reassert eval.
    rows=list(runtime.iter_generate(records,batch_size=2))
    h1.remove();h2.remove()
    assert len(encoded)==len(decoded)==1
    assert torch.equal(encoded[0],decoded[0])
    assert not torch.allclose(encoded[0][0],encoded[0][1])
    assert runtime.extractor.calls==[['A'*10,'C'*10]]
    assert [r['template_id'] for r in rows]==['a','c']
    assert all(not p.requires_grad and p.grad is None for p in runtime.model.parameters())


@pytest.mark.parametrize('length',[None,10,80,150])
@pytest.mark.parametrize('samples',[1,3])
def test_default_explicit_lengths_and_argmax_repeats(runtime,length,samples):
    rows=runtime.generate_from_template('ACDEFGHIKLAC',target_length=length,num_samples=samples)
    assert len(rows)==samples and [r['sample_index'] for r in rows]==list(range(samples))
    assert all(len(r['sequence'])==(12 if length is None else length) for r in rows)
    assert len({r['sequence'] for r in rows})==1
    assert all(set(r['sequence'])<=set(AA_ORDER) for r in rows)


def test_batched_templates_reproducible_sampling_advances_private_rng(runtime):
    torch.manual_seed(932)
    before=torch.get_rng_state().clone()
    settings=dict(num_samples=4,temperature=1.,seed=57,batch_size=2)
    first=list(runtime.iter_generate(templates(),**settings))
    second=list(runtime.iter_generate(templates(),**settings))
    assert first==second and torch.equal(before,torch.get_rng_state())
    assert [r['target_length'] for r in first]==[10]*4+[23]*4+[150]*4
    assert len({r['sequence'] for r in first[:4]})>1
    other=list(runtime.iter_generate(templates(),**{**settings,'seed':58}))
    assert [r['sequence'] for r in other]!=[r['sequence'] for r in first]
    assert len(runtime.extractor.calls)==6  # two batches in each of three calls


@pytest.mark.parametrize('kwargs',[
    {'target_length':9},{'target_length':151},{'target_length':50.0},{'target_length':True},
    {'num_samples':0},{'num_samples':-1},{'num_samples':True},
    {'temperature':-1},{'temperature':float('nan')},{'temperature':float('inf')},
    {'temperature':'1'},{'temperature':True},{'seed':-1},{'seed':2**63},{'seed':False},
])
def test_bad_parameters_rejected_before_esm(runtime,kwargs):
    with pytest.raises(ValueError):runtime.generate_from_template('A'*10,**kwargs)
    assert not runtime.extractor.calls


@pytest.mark.parametrize('sequence',['A'*9,'A'*151,'X'*10,'a'*10,'ACD EFGHIKL'])
def test_invalid_template_is_rejected_not_filtered(runtime,sequence):
    with pytest.raises(ValueError):runtime.generate_from_template(sequence)
    assert not runtime.extractor.calls


def test_nonempty_conditions_are_explicit(runtime,tmp_path):
    with pytest.raises(NotImplementedError):runtime.generate_from_template('A'*10,condition={})
    with pytest.raises(NotImplementedError):runtime.encode_records(templates(),condition='x')
    with pytest.raises(NotImplementedError):
        export_raw_latents(runtime,templates(),tmp_path/'bad',expected_records=3,condition={})
    assert not (tmp_path/'bad').exists()


def test_extreme_positive_temperature_and_nonfinite_logits():
    scores=torch.arange(20).float().repeat(10,1)
    assert (sample_logits(scores,1e-320,torch.Generator().manual_seed(1))==19).all()
    assert sample_logits(scores,1e300,torch.Generator().manual_seed(1)).shape==(10,)
    with pytest.raises(ValueError):sample_logits(torch.full((10,20),float('nan')),1,torch.Generator())


def test_candidates_traceability_duplicates_and_no_overwrite(runtime,tmp_path):
    result=write_candidates(runtime,templates(),tmp_path/'out',expected_records=3,num_samples=3,temperature=0.,batch_size=2)
    assert result['candidate_count']==9
    assert result['within_template_duplicate_rate']==pytest.approx(2/3)
    assert all(x['duplicate_rate']==pytest.approx(2/3) for x in result['per_template'].values())
    rows=[json.loads(x) for x in (tmp_path/'out/metadata.jsonl').read_text().splitlines()]
    assert len({r['candidate_id'] for r in rows})==9
    assert rows[0]['template_sequence_hash']==templates()[0].sequence_hash
    assert rows[3]['template_id']==rows[0]['template_id']  # repeated IDs remain distinct via index
    fasta=list(selected_records(tmp_path/'out/candidates.fasta',None))
    assert [r.sequence for r in fasta]==[r['sequence'] for r in rows]
    with pytest.raises(FileExistsError):write_candidates(runtime,templates(),tmp_path/'out',expected_records=3)


def test_latent_export_shards_raw_repeatable_and_reload_decodes(runtime,tmp_path):
    items=templates()+[PeptideRecord('four','D'*11),PeptideRecord('five','E'*12)]
    first=export_raw_latents(runtime,iter(items),tmp_path/'first',expected_records=5,shard_size=2,batch_size=2)
    second=export_raw_latents(runtime,iter(items),tmp_path/'second',expected_records=5,shard_size=2,batch_size=2)
    assert [s['count'] for s in first['shards']]==[2,2,1]
    assert first['normalization'] is None and first['latent_shape']==[50,32]
    a=list(iter_latent_records(tmp_path/'first',expected_identity=runtime.identity))
    b=list(iter_latent_records(tmp_path/'second',expected_identity=runtime.identity))
    # Raw saved tensors equal direct encoder outputs, without centering/scaling.
    direct=[]
    for records,latent in runtime.iter_encoded(items,2):direct.extend(latent.unbind())
    for index,(left,right,raw) in enumerate(zip(a,b,direct)):
        assert left['record_index']==index
        assert torch.equal(left['latent'],right['latent']) and torch.equal(left['latent'],raw)
        assert left['latent'].shape==(50,32) and not left['latent'].requires_grad
    latent=torch.stack([r['latent'] for r in a]);lengths=torch.tensor([r.length for r in items])
    loaded=runtime.decode_latents(latent,lengths)['logits']
    original=runtime.decode_latents(torch.stack(direct),lengths)['logits']
    assert torch.equal(loaded,original)
    assert len(runtime.extractor.calls)==9  # only small batches, never all latents at once


@pytest.mark.parametrize('damage',['file','count','source','record_hash','shape','normalization'])
def test_latent_corruption_and_wrong_source_rejected(runtime,tmp_path,damage):
    export_raw_latents(runtime,templates(),tmp_path/'export',expected_records=3,shard_size=2)
    manifest_path=tmp_path/'export/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    path=tmp_path/'export'/manifest['shards'][0]['file']
    if damage=='file':path.write_bytes(b'broken')
    elif damage=='count':manifest['record_count']+=1
    elif damage=='source':manifest['identity']['checkpoint_sha256']='b'*64
    elif damage=='normalization':manifest['normalization']={'mean':0}
    else:
        payload=torch.load(path,weights_only=True)
        if damage=='record_hash':payload['records'][0]['sequence_hash']='wrong'
        if damage=='shape':payload['records'][0]['latent']=torch.zeros(1,32)
        torch.save(payload,path)
        manifest['shards'][0]['sha256']=file_sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):list(iter_latent_records(tmp_path/'export',expected_identity=runtime.identity))


def test_fasta_strict_preflight_and_limit(tmp_path):
    path=tmp_path/'x.fa';path.write_text('>good\nACDEFGHIKL\n>bad\nXXXXXXXXXX\n')
    assert inspect_fasta(path,1)['selected_records']==1
    with pytest.raises(ValueError,match='Nonstandard'):inspect_fasta(path,None)
    with pytest.raises(ValueError):inspect_fasta(path,0)
    path.write_text('')
    with pytest.raises(ValueError,match='No templates'):inspect_fasta(path,3)


@pytest.mark.parametrize('module',['generate','reconstruct','export_latents'])
def test_help_has_no_heavy_import_or_file_side_effects(tmp_path,module):
    code='''
import sys,runpy,importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','esm','yaml','raygun'}:raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
sys.argv=[sys.argv[1],'--help']
runpy.run_module(sys.argv[0],run_name='__main__')
'''
    result=subprocess.run([sys.executable,'-I','-c',code,'templatedf.'+module],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '--checkpoint' in result.stdout and not list(tmp_path.iterdir())


@pytest.mark.parametrize('module,args',[('generate',['--temperature','-1']),('generate',['--target-length','151']),
                                      ('generate',['--num-samples','0']),('export_latents',['--shard-size','0']),
                                      ('reconstruct',['--batch-size','0'])])
def test_invalid_cli_fails_before_opening_model_or_input(tmp_path,module,args):
    command=[sys.executable,'-m','templatedf.'+module,'--checkpoint',str(tmp_path/'missing.pt'),
             '--fasta',str(tmp_path/'missing.fa'),'--output-dir',str(tmp_path/'out'),'--limit','5',*args]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==2 and 'FileNotFoundError' not in result.stderr
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('damage',['aa','stage','config','source'])
def test_checkpoint_identity_fails_before_model_or_esm_construction(tmp_path,monkeypatch,damage):
    from dataclasses import asdict
    from templatedf.feature_cache import CacheSpec
    import templatedf.checkpoint as checkpoints
    import templatedf.esm_features as esm
    cfg={'dim':1280,'num_latents':50}
    spec=CacheSpec(source='test:trusted-source')
    state={'stage':'1A','aa_order':AA_ORDER,'config':{'model':dict(cfg),'esm':{'checkpoint_path':'esm.pt','contact_regression_path':'reg.pt'}},
           'model_config':dict(cfg),'esm_source':asdict(spec),'step':300}
    if damage=='aa':state['aa_order']='wrong'
    if damage=='stage':state['stage']='1B'
    if damage=='config':state['model_config']['num_latents']=49
    if damage=='source':state['esm_source']['source']='wrong'
    monkeypatch.setattr(checkpoints,'load_checkpoint',lambda path:state)
    monkeypatch.setattr(esm,'local_cache_spec',lambda *a,**k:spec)
    def fail(*a,**k):raise AssertionError('ESM must not load with incompatible checkpoint')
    monkeypatch.setattr(esm.ESMFeatureExtractor,'from_local',fail)
    path=tmp_path/'checkpoint.pt';path.write_bytes(b'fixture: loader injected')
    with pytest.raises(ValueError):TemplateGenerator.from_checkpoint(path)
