import ast,importlib.metadata,json,os,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

class PreflightTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parent
        tree=ast.parse((root/'lora_trainer_gui.py').read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LoRATrainerGUI')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_start_training_launch')
        self.work=[];self.callbacks=[]
        self.sub=SimpleNamespace(run=Mock(return_value=SimpleNamespace(returncode=0,stdout=json.dumps({'has_cuda':True,'vram_free_gb':14,'vram_gb':16,'is_rocm':True,'gcn_arch':'gfx1030'}),stderr='')),CREATE_NO_WINDOW=0)
        class Thread:
            def __init__(inner,target,daemon):inner.target=target
            def start(inner):self.work.append(inner.target)
        env={'os':os,'subprocess':self.sub,'threading':SimpleNamespace(Thread=Thread),'json':json,'FIZGIG_DIR':str(root),'tk':SimpleNamespace(DISABLED='disabled',NORMAL='normal'),'ARCHITECTURES':{'Krea 2':{'is_krea2':True}}}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<preflight>','exec'),env)
        self.launch=env['_start_training_launch']
        self.gui=SimpleNamespace(architecture_var=SimpleNamespace(get=lambda:'Krea 2'),_start_training_btn=Mock(),update_console=Mock(),_venv_python=lambda:'python',_cuda_env_for_subprocess=lambda env:env,master=SimpleNamespace(after=lambda delay,cb:self.callbacks.append(cb)),_start_training_launch_ready=Mock())

    def begin(self):
        with patch.object(importlib.metadata,'version',return_value='2.12+rocm7'),patch.object(os,'name','nt'):
            self.launch(self.gui)

    def test_launch_returns_before_query_and_finishes_on_tk_poll(self):
        self.begin();self.sub.run.assert_not_called();self.gui._start_training_launch_ready.assert_not_called()
        self.begin();self.assertEqual(len(self.work),1)
        self.work.pop()();self.gui._start_training_launch_ready.assert_not_called()
        self.callbacks.pop()()
        self.gui._start_training_launch_ready.assert_called_once()
        self.assertIsNone(self.gui._training_gpu_caps)
        self.assertFalse(self.gui._training_start_pending)

    def test_cancel_does_not_launch_after_worker_finishes(self):
        self.begin();self.gui._gpu_preflight_cancelled=True
        self.work.pop()();self.callbacks.pop()()
        self.gui._start_training_launch_ready.assert_not_called()

    def test_failed_query_does_not_retry_on_gui_thread(self):
        self.sub.run.side_effect=RuntimeError('timeout')
        self.begin();self.work.pop()();self.callbacks.pop()()
        self.gui._start_training_launch_ready.assert_not_called()
        self.assertFalse(self.gui._training_start_pending)

    def test_auto_validation_does_not_resolve_gpu_strategy(self):
        source=(Path(__file__).resolve().parent/'lora_trainer_gui.py').read_text(encoding='utf-8')
        tree=ast.parse(source)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LoRATrainerGUI')
        validate=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='validate_inputs')
        block=next(n for n in validate.body if isinstance(n,ast.Try) and '_swap_raw' in ast.unparse(n))
        gui=SimpleNamespace(entries={'BLOCKS_SWAP':SimpleNamespace(get=lambda:'Auto (detect from GPU)')},_parse_blocks_swap=Mock(side_effect=AssertionError('GPU probe')))
        env={'self':gui,'errors':[],'config':{'blocks_swap_max':26},'arch':'Krea 2'}
        exec(compile(ast.Module(body=[block],type_ignores=[]),'<validation>','exec'),env)
        gui._parse_blocks_swap.assert_not_called()
        self.assertEqual(env['errors'],[])

    def test_auto_precision_is_resolved_with_manual_swap_at_launch(self):
        tree=ast.parse((Path(__file__).resolve().parent/'lora_trainer_gui.py').read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LoRATrainerGUI')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_resolve_krea2_training_swap')
        env={}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<resolve>','exec'),env)
        for nf4,int8,expected in ((True,'',0),(False,'bf16',0),(False,'',16)):
            gui=SimpleNamespace(entries={'BLOCKS_SWAP':SimpleNamespace(get=lambda:'16 (Max)')},
                _base_precision=lambda:'auto', _auto_krea2_strategy=Mock(),
                quant_4bit_var=SimpleNamespace(get=lambda:nf4),_auto_quant_int8=int8,
                _parse_blocks_swap=Mock(return_value=16),update_console=Mock())
            self.assertEqual(env[fn.name](gui),expected)
            gui._auto_krea2_strategy.assert_called_once()
            if expected == 0:
                gui._parse_blocks_swap.assert_not_called()
        gui._base_precision=lambda:'fp8'
        gui._auto_krea2_strategy.reset_mock()
        self.assertEqual(env[fn.name](gui),16)
        gui._auto_krea2_strategy.assert_not_called()

    def test_metadata_detection_never_runs_kernel_probes(self):
        from fizgig.utils import capabilities as caps
        fake=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,
            get_device_properties=lambda i:SimpleNamespace(name='RX 6800',gcnArchName='gfx1030',total_memory=16*2**30),
            get_device_capability=lambda i:(10,3),mem_get_info=lambda i:(14*2**30,16*2**30)))
        caps.detect.cache_clear()
        with patch.dict(sys.modules,{'torch':fake}),patch.object(caps,'is_rocm',return_value=True),patch.object(caps,'_probe_scaled_mm') as scaled,patch.object(caps,'_probe_int_mm') as integer,patch('importlib.util.find_spec',return_value=object()):
            for options in ({'probe_kernels':False}, {}):
                result=caps.detect(**options)
                scaled.assert_not_called();integer.assert_not_called()
                self.assertTrue(result.bitsandbytes)
                self.assertEqual(result.vram_free_gb,14)
        caps.detect.cache_clear()

    def test_incompatible_int8_swap_is_rejected_before_launch(self):
        source=(Path(__file__).resolve().parent/'lora_trainer_gui.py').read_text(encoding='utf-8')
        tree=ast.parse(source)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LoRATrainerGUI')
        validate=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='validate_inputs')
        block=next(n for n in validate.body if isinstance(n,ast.Try) and '_swap_raw' in ast.unparse(n))
        gui=SimpleNamespace(entries={'BLOCKS_SWAP':SimpleNamespace(get=lambda:'16 (Max)')},_parse_blocks_swap=lambda:16,_base_precision=lambda:'int8')
        env={'self':gui,'errors':[],'config':{'blocks_swap_max':26,'is_krea2':True},'arch':'Krea 2'}
        exec(compile(ast.Module(body=[block],type_ignores=[]),'<validation>','exec'),env)
        self.assertEqual(len(env['errors']),1)
        self.assertIn('INT8 cannot use block swap',env['errors'][0])

if __name__=='__main__':
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
    unittest.main()
