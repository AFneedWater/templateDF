"""Read-only stage 05 checkpoint restore and independent validation replay."""
import json
from pathlib import Path
import torch
import yaml
from run_pilot import ART, LOG, datasets, measured_engine_class, dump
from templatedf.data import file_sha256


def numeric_differences(left,right,path=''):
    result={}
    for key,value in left.items():
        target=right[key]
        name=f'{path}.{key}' if path else key
        if isinstance(value,dict):
            result.update(numeric_differences(value,target,name))
        elif isinstance(value,(int,float)):
            result[name]=abs(value-target)
        else:
            assert value==target
    return result


def main():
    cfg=yaml.safe_load((ART/'ce_config.yaml').read_text())
    recorded=json.loads((ART/'ce_run/summary.json').read_text())
    checkpoint=ART/'ce_run/last.pt'
    before_hash=file_sha256(checkpoint)
    assert before_hash==recorded['checkpoint_sha256']
    ds,provenance=datasets()
    engine=measured_engine_class()(cfg,ds['train'],ds['validation'],data_provenance=provenance,resume=checkpoint)
    assert engine.step==300 and engine.settings.max_steps==300
    steps=sorted({int(state['step']) for state in engine.optimizer.state.values()})
    assert steps==[300]
    with torch.no_grad():
        actual=engine.evaluate()
    differences=numeric_differences(recorded['final_validation'],actual)
    assert max(differences.values())<1e-6
    budget_stopped=False
    try:
        engine.train_step()
    except RuntimeError as exc:
        if 'max_steps budget exhausted' not in str(exc):
            raise
        budget_stopped=True
    assert budget_stopped and engine.step==300
    assert file_sha256(checkpoint)==before_hash
    report={'checkpoint_sha256':before_hash,'step':engine.step,'optimizer_step_values':steps,
            'optimizer_state_entries':len(engine.optimizer.state),'sampler_epoch':engine.sampler.epoch,
            'sampler_cursor':engine.sampler.cursor,'sampler_size':engine.sampler.size,
            'saved_validation_replayed':actual,'validation_max_absolute_difference':max(differences.values()),
            'all_metric_differences':differences,'additional_optimizer_steps':0,
            'exhausted_budget_rejected':budget_stopped,'checkpoint_unchanged':True}
    dump(LOG/'checkpoint_verification.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
