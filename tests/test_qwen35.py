"""CPU-only architecture and checkpoint tests: no pretrained weights or training."""
import pytest


def test_qwen35_text_checkpoint_extraction_and_hybrid_lora(tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from transformers import Qwen3_5Config,Qwen3_5TextConfig,Qwen3_5VisionConfig,Qwen3_5ForConditionalGeneration
    from financeqa.modeling import load_backbone
    from peft import get_peft_model,LoraConfig
    text=Qwen3_5TextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,head_dim=16,
        linear_num_key_heads=2,linear_num_value_heads=2,linear_key_head_dim=8,linear_value_head_dim=8,
        layer_types=['linear_attention','full_attention'],max_position_embeddings=128,
        eos_token_id=2,pad_token_id=0)
    vision=Qwen3_5VisionConfig(depth=1,hidden_size=32,intermediate_size=64,num_heads=2,
        out_hidden_size=32,patch_size=2,temporal_patch_size=1,num_position_embeddings=16,
        deepstack_visual_indexes=[])
    full=Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=text.to_dict(),vision_config=vision.to_dict(),
        image_token_id=60,video_token_id=61,vision_start_token_id=62,vision_end_token_id=63))
    expected=full.model.language_model.layers[0].linear_attn.in_proj_qkv.weight.detach().clone()
    full.save_pretrained(tmp_path)
    model=load_backbone({'name':str(tmp_path),'revision':'main','loader':'qwen3_5_text',
                         'device':'cpu','dtype':'float32','attn_implementation':'sdpa'})
    assert model.__class__.__name__=='Qwen3_5ForCausalLM'
    assert not any('visual' in n for n,_ in model.named_parameters())
    assert torch.equal(model.model.layers[0].linear_attn.in_proj_qkv.weight,expected)
    assert torch.equal(model.lm_head.weight,full.lm_head.weight)
    peft=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=2,lora_alpha=4,target_modules='all-linear'))
    names=[n for n,p in peft.named_parameters() if p.requires_grad]
    assert any('linear_attn.in_proj_qkv' in n for n in names)
    assert any('self_attn.q_proj' in n for n in names)
    assert any('mlp.up_proj' in n for n in names)
    assert not any('embed_tokens' in n or 'lm_head' in n for n in names)


def test_chat_settings_reject_thinking_mode():
    from financeqa.modeling import chat_kwargs
    assert chat_kwargs({})=={'enable_thinking':False}
    with pytest.raises(ValueError,match='enable_thinking=false'):
        chat_kwargs({'chat_template_kwargs':{'enable_thinking':True}})


def test_real_cached_qwen35_tokenizer_mask():
    pytest.importorskip('transformers')
    from pathlib import Path
    from transformers import AutoTokenizer
    from financeqa.modeling import DEFAULT_REVISION
    from train.sft import tokenize_example
    root=Path(__file__).resolve().parents[1]
    snapshot=root/'.cache/huggingface/models--Qwen--Qwen3.5-2B/snapshots'/DEFAULT_REVISION
    if not (snapshot/'tokenizer.json').exists():
        pytest.skip('Official tokenizer metadata not cached; tests never download it')
    tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
    row={'messages':[{'role':'system','content':'Return JSON.'},{'role':'user','content':'Growth from 100 to 120?'},
                     {'role':'assistant','content':'{"answer":20,"unit":"percent"}'}]}
    enc=tokenize_example(row,tok,512,{'enable_thinking':False})
    target=tok.decode([t for t in enc['labels'] if t!=-100])
    assert target=='{"answer":20,"unit":"percent"}<|im_end|>\n'
    assert tok.eos_token_id in enc['labels']
    prompt=tok.decode(enc['input_ids'][:next(i for i,x in enumerate(enc['labels']) if x!=-100)])
    assert '<think>\n\n</think>' in prompt


def test_trainer_interfaces_without_training(tmp_path):
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    pytest.importorskip('trl')
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast,Qwen3_5TextConfig,Qwen3_5ForCausalLM,Trainer,TrainingArguments
    from peft import get_peft_model,LoraConfig
    from datasets import Dataset
    from trl import GRPOConfig,GRPOTrainer
    from rewards.numerical import numerical_reward
    from train.grpo import align_reference_adapter
    cfg=Qwen3_5TextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,head_dim=16,
        linear_num_key_heads=2,linear_num_value_heads=2,linear_key_head_dim=8,linear_value_head_dim=8,
        layer_types=['linear_attention','full_attention'],max_position_embeddings=128,
        eos_token_id=2,pad_token_id=0)
    model=get_peft_model(Qwen3_5ForCausalLM(cfg),LoraConfig(task_type='CAUSAL_LM',r=2,lora_alpha=4,target_modules='all-linear'))
    raw=Tokenizer(WordLevel({'[PAD]':0,'[BOS]':1,'[EOS]':2,'[UNK]':3,'question':4},unk_token='[UNK]'))
    raw.pre_tokenizer=Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=raw,pad_token='[PAD]',bos_token='[BOS]',eos_token='[EOS]',unk_token='[UNK]')
    tok.chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}"
    sft_args=TrainingArguments(output_dir=str(tmp_path/'sft'),use_cpu=True,bf16=False,
        warmup_steps=.05,report_to='none',per_device_train_batch_size=1,gradient_accumulation_steps=8)
    sft=Trainer(model=model,args=sft_args,processing_class=tok)
    assert sft.state.global_step==0
    args=GRPOConfig(output_dir=str(tmp_path/'grpo'),use_cpu=True,bf16=False,max_steps=1,
        gradient_checkpointing=False,report_to='none',per_device_train_batch_size=1,gradient_accumulation_steps=8,
        num_generations=8,beta=.04,disable_dropout=True,chat_template_kwargs={'enable_thinking':False})
    ds=Dataset.from_list([{'prompt':[{'role':'user','content':'question'}],
                          'reference':{'value':.2,'unit':'ratio'},'id':'fixture'}])
    trainer=GRPOTrainer(model=model,args=args,processing_class=tok,reward_funcs=numerical_reward,train_dataset=ds)
    assert trainer.state.global_step==0  # Constructor only; neither .train nor forward/backward is called.
    assert align_reference_adapter(model)['initial_reference_equals_sft']


def test_render_chat_normalizes_transformers5_mapping():
    from financeqa.modeling import render_chat
    class MappingTokenizer:
        def apply_chat_template(self,*args,**kwargs):
            return {'input_ids':[1,2,3],'attention_mask':[1,1,1]}
    assert render_chat(MappingTokenizer(),[],generation=True)==[1,2,3]
