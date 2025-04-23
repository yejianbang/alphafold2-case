import torch
from alphafold.common import protein
from alphafold.model import config
from alphafold_pytorch_jit import net as model
import jax
import numpy as np
import os
import time 
import pickle
from runners.timmer import Timmers

try:
  use_tpp = (os.environ.get('USE_TPP') == '1')
  if use_tpp:
    from alphafold_pytorch_jit.basics import GatingAttention
    from tpp_pytorch_extension.alphafold.Alpha_Attention import GatingAttentionOpti_forward
    GatingAttention.forward = GatingAttentionOpti_forward
    from alphafold_pytorch_jit.backbones import TriangleMultiplication
    from tpp_pytorch_extension.alphafold.Alpha_TriangleMultiplication import TriangleMultiplicationOpti_forward
    TriangleMultiplication.forward = TriangleMultiplicationOpti_forward
    is_tpp = True
    print('Running with Intel Optimizations. TPP extension detected.')
  else:
    is_tpp = False
    print('[warning] No TPP extension detected')
except:
  is_tpp = False
  print('[warning] No TPP extension detected, will fallback to imperative mode')

def main():
    
    bf16 = (os.environ.get('AF2_BF16') == '1')
    print("bf16 variable: ", bf16)
    data =  np.load('/your_open-omics-alphafold_dir/af_output/T1050/intermediates/processed_features.npz', allow_pickle=True)
    processed_feature_dict= {}
    for k in data.files:
        processed_feature_dict[k] = data[k]
    processed_feature_dict.keys()

    plddts = {}

    processed_feature_dict = jax.tree_map(
        lambda x:torch.tensor(x), processed_feature_dict)
        

    from runners.timmer import Timmers
    h_timmer = Timmers('/your_open-omics-alphafold_dir/af_output/timmers_T1050.txt')

    num_ensemble = 1
    random_seed = 0

    model_runners = {}
    model_list = ['model_1_ptm']
    print("List of models:", model_list)
    for model_name in model_list:
        model_config = config.model_config(model_name)
        model_config['data']['eval']['num_ensemble'] = num_ensemble
        root_params = '/your_open-omics-alphafold_dir/weights/extracted/' + model_name
        model_runner = model.RunModel(
        model_config, 
        root_params, 
        h_timmer,
        random_seed)
        model_runners[model_name] = model_runner

    model_runner = model_runners['model_1_ptm']

    h_timmer.add_timmer('model_1_ptm infer')
    
    with torch.no_grad():
        with torch.cpu.amp.autocast(enabled=bf16):
            prediction_result = model_runner(processed_feature_dict)

    print('### [INFO] post-assessment: plddt')
    timmer_name = f'post-assessment by plddt: {model_name}'

    plddts[model_name] = np.mean(prediction_result['plddt'])
    print("plddts score = ", plddts[model_name])

if __name__ == '__main__':
    prof =  torch.profiler.profile(
       activities=[torch.profiler.ProfilerActivity.CPU],
       schedule=torch.profiler.schedule(wait=0,warmup=0,active=1,repeat=1),
       on_trace_ready=torch.profiler.tensorboard_trace_handler("log_tensor","intel_0629"),
       record_shapes=True,
       profile_memory=False,
       with_stack=False,
       use_cuda=False
    ) 

    prof.start()
    t1 = time.time();
    main()
    t2 = time.time();
    print('  # [TIME] tota duration =', (t2 - t1), 'sec')
    prof.stop()
