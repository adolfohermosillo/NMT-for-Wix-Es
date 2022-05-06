import argparse
import os
import script
from sacrebleu import BLEU, CHRF
import numpy as np
argp = argparse.ArgumentParser()

#
# Model input
#

corpus = argp.add_argument('--corpus',default = "",
                            help='Select corpus')

source = argp.add_argument('--source',default = "",
                           help='Select Source language')

target = argp.add_argument('--target',default = "",
                            help='Select Target language')


#
# Architecture
#

transformer_layers =  argp.add_argument('--transformer_layers',
                            default = 6,
                           help='Set number of encoder/decoder layers')

d_model = argp.add_argument('--d_model',
                            default = 512,
                           help='Set dimension for embeddings')

attention_heads = argp.add_argument('--attention_heads',
                            default = 8,
                           help='Set number of attention heads')

FFN =  argp.add_argument('--FFN',
                            default = 1024,
                           help='Set number of attention heads')

vocab_size = argp.add_argument('--vocab_size',
                        default = '',
                        help='Select vocabulary size from [ 500, 1000,2000, 3000, 4000, 5000]')

#
# Training
#
early_stopping = argp.add_argument('--early_stopping',
                            default = "10",
                           help='Set number of validation steps for early stopping')

early_stopping_criteria = argp.add_argument('--early_stopping_criteria',
                            default = "accuracy",
                           help='Select early stopping criteria from ["ppl", "accuracy"]')

dropout =  argp.add_argument('--dropout',
                            default = 0.3,
                           help='Set number of attention heads')

attention_dropout =  argp.add_argument('--attention_dropout',
                            default = 0.3,
                           help='Set number of attention heads')

learning_rate = argp.add_argument('--learning_rate',
                            default = 2,
                           help='Set learning_rate')

warmup_steps = argp.add_argument('--warmup_steps',
                            default = 8000,
                           help='Set number of attention heads')

subword_regularization = argp.add_argument('--subword_regularization',
                                help='Select tokenizer from [True, False]')
batch_size = argp.add_argument('--batch_size',
                            default = 32,
                           help='Set number of examples per batch')

l_sample = argp.add_argument('--l_sample',
                                default = 32,
                                help='Select number of sample for unigram tokenization, ignored for bpe')
a_prob = argp.add_argument('--a_prob',
                                default = 0.1,
                                help='Sets smoothing paramater for suwbord regularization or probability for bpe dropout')

valid_steps = argp.add_argument('--valid_steps',
                            default = "1000",
                           help='Set validation every X steps')

iterations = argp.add_argument('--iterations',default='',
                                help="Select number of iterations to train")


#
# Tokenization
#

tokenizer = argp.add_argument('--tokenizer',
                                help='Select tokenizer from [unigram, bpe, morf, char, word]')

transforms = argp.add_argument('--transforms',
                            default = '[filtertoolong,sentencepiece]',
                            help='Select Target language')


#
# Experiment specific parameters
#
seed = argp.add_argument('--seed',
                          default = 1,
                           help='Set random seed used for reproducibility')

run = argp.add_argument('--run',
                          default = '',
                           help='Set for the number of runs, should also have a different seed')

mode = argp.add_argument('--mode', default = 'train',
                            help='Select [train, translate, evaluate ]')

debug = argp.add_argument('--debug', default = 'True',
                            help='Select [True,False]')

#
# Inference
#

step = argp.add_argument('--step', default = "",
                           help='Set step to use for inference')


set_type = argp.add_argument('--set_type', default = 'dev',
                           help='Set step to use for inference')
evaluate = argp.add_argument('--evaluate', default = 'False',
			   help='Select to translate dev data to find best model')


gpu = argp.add_argument('--gpu', default = '0',
                           help='Set step to use for inference')

args = argp.parse_args()

# abbreviations for extensions
abbr = {"spanish" : 'es', 'wixarika': 'hch', 'english':'en', 'sepidi':'nso'}


#set directory to current working directory
directory = (os.getcwd())


assert args.tokenizer != '', 'No tokenizer provided'
assert args.vocab_size != '', 'No vocab_size provided'
assert args.corpus != '', 'No corpus provided'
assert args.source != '', 'No source provided'
assert args.target != '', 'No target provided'
assert args.run != '', 'No run provided'

src = abbr[args.source]
tgt = abbr[args.target]

#set whether the model will use subword regularization
regularized = 'no_subword_regularized'
subword_regularization_script = ''
if args.subword_regularization == 'True':
    regularized = 'subword_regularized'
    subword_regularization_script = script.subword_regularization.format(args.l_sample,args.a_prob,args.l_sample,args.a_prob )


xmodel = f"models/{args.tokenizer}/to{args.target}/{args.vocab_size}/{regularized}"
# where the models will be saved
model_path = f'{directory}/{xmodel}'

# path to training data
data_path = f'{directory}/data/{args.corpus}'
train_path =  f'{data_path}/train'
dev_path =  f'{data_path}/dev'
test_path =  f'{data_path}/test'


# path to vocabularies
vocab_path = f'{model_path}/vocabulary/vocab'

# path to tokenizers
tokenizers_path = f'{directory}/tokenizers'

if args.tokenizer in ['char','morf','word']:
    args.vocab_size = args.tokenizer

if args.tokenizer == 'morf':
    args.transforms = '[filtertoolong]'
    tokenizers_scripts = ''
else :
    tokenizers_scripts = script.tokenizers.format(f'{tokenizers_path}/{args.tokenizer}/{args.source}.{args.vocab_size}.model',
                                            f'{tokenizers_path}/{args.tokenizer}/{args.target}.{args.vocab_size}.model')




# where the model will be saved
save_model = script.save.format(model_path)




# where the data will be accessed
data_script = script.data.format(train_path+f'.{src}',train_path+f'.{tgt}',
                   args.transforms, dev_path+f'.{src}',dev_path+f'.{tgt}',
                   args.transforms)




# where the vocab files will be accessed
vocab_scripts = script.vocab.format(f'{vocab_path}.src', f'{vocab_path}.tgt')






# setting model hyperparamters
model_script = script.model.format(args.transformer_layers, args.transformer_layers,
                                    args.attention_heads, args. d_model,
                                    args.d_model, args.FFN, args.dropout, args.attention_dropout)






# path to where the logs will be written
log = model_path+f'/log/log_{args.run}'
log_script = script.logging.format(log)




# Set early stopping criteria
early_stopping_script = script.early_stopping.format(args.early_stopping, args.early_stopping_criteria)
general_script = script.general.format(f'{model_path}/steps/{args.run}', args.valid_steps, args.valid_steps, args.iterations )



optimization_script = script.optimization.format(args.learning_rate,args.warmup_steps)
batching_script = script.batching.format(args.batch_size)

# create config file
config = (save_model, vocab_scripts, data_script, tokenizers_scripts, subword_regularization_script,
         script.pyonmttok, general_script, script.reproducibility.format(args.seed), early_stopping_script,
           batching_script, optimization_script, model_script,log_script)


if args.mode == "train" :

    if args.debug == 'True':

        print("".join([i for i in config]))
        print(f'onmt_build_vocab -config {model_path}/config.yaml -n_sample -1')
        print(f'onmt_train -config {model_path}/config.yaml')

    else:

        with open (f'{model_path}/config.yaml', 'w') as f:
            f.write("".join([i for i in config]))

        try:
            os.system(f'onmt_build_vocab -config {model_path}/config.yaml -n_sample -1')
        except:
            pass
        os.system(f'onmt_train -config {model_path}/config.yaml')


if args.mode == "translate" :
    assert args.set_type != "", 'No set_type provided'
    model = f'{model_path}/steps/{args.run}_step_'
    src_path= f'{data_path}/encoded/{args.tokenizer}.{args.vocab_size}.{args.set_type}.{src}'
    output_path= f'{model_path}/predictions/encoded/{args.run}.{args.corpus}.{args.set_type}.'
    encoded_output = f'{xmodel}/predictions/encoded/{args.run}.{args.corpus}.{args.set_type}.'
    decoded_output =  f'{xmodel}/predictions/decoded/{args.run}.{args.corpus}.{args.set_type}.'
    if args.evaluate == 'True':
        assert args.iterations  != '', 'No iterations provided'
        for iter in range(1000, int(args.iterations)+1, 1000 ):
            mod = model+f'{iter}.pt'
            command = f'onmt_translate -model {mod} -src {src_path} -output {output_path}{iter} -gpu {args.gpu}'

            if os.path.exists(mod):
                if args.debug == 'True':
                    print(command)
                else:
                    print(command)
                    os.system(command)
                    dec = f"""python tokenizer.py --input_path {encoded_output}{iter} --model_directory tokenizers \
                            --model_output {decoded_output}{iter} --language {args.target} \
                            --vocab_size {args.vocab_size} --mode decode --model_type {args.tokenizer}"""
                    os.system(dec)

            else:
                print(f'Stopped at {iter}')
                break
    else:
        assert args.step != "", 'No step provided'
        model = model+f'{args.step}.pt'
        command = f'onmt_translate -model {model} -src {src_path} -output {output_path}{args.step} -gpu {args.gpu}'
        if os.path.exists(model):
            if args.debug == 'True':
                print(command)
            else:
                os.system(command)
        else:
            print(f'Model: {model} is not saved')

if args.mode == "evaluate" :

    assert args.iterations  != '', 'No iterations provided'
    assert args.set_type != "dev", 'set_type should not be the default "dev"'
    references = dev_path+f'.{tgt}'
    decoded_output =  f'{model_path}/predictions/decoded/{args.run}.{args.corpus}.dev.'



    bleu = BLEU()
    chrf = CHRF()
    refs = [open(references,'r').readlines()]
    blue_scores = []
    chrf_scores = []
    print('iteration','bleu','chrf')
    for iter in range(1000, int(args.iterations)+1, 1000 ):
        hypothesis = f"{decoded_output}{iter}"
        sys = open(hypothesis,'r').readlines()
        bl = bleu.corpus_score(sys, refs).score
        cr = chrf.corpus_score(sys, refs).score
        blue_scores.append(bl)
        chrf_scores.append(ch)



    #get best perfroming model according to bleu
    best_bleu_model = np.argmax(blue_scores)
    best_bleu_score = blue_scores[best_bleu_model]
    print(f'Best iteration according to chrf: {best_chrf_model+1}000, with a bleu score of {best_bleu_score}')

    #get best perfroming model according to chrf
    best_chrf_model = np.argmax(chrf_scores)
    best_chrf_score = blue_scores[best_chrf_model]
    print(f'Best iteration according to chrf: {best_chrf_model+1}000, with a chrf score of {best_chrf_score}')
    #evaluate on test set

    model_bleu = f'{model_path}/steps/{args.run}_step_{best_bleu_model+1}000'
    model_chrf = f'{model_path}/steps/{args.run}_step_{best_chrf_model+1}000'

    #output_path = f'{model_path}/predictions/encoded/{args.run}.{args.corpus}.{args.set_type}.'
    #output_path =  f'{model_path}/predictions/encoded/{args.run}.{args.corpus}.{args.set_type}.'


    command = 'onmt_translate -model {} -src {} -output {}{} -gpu {}'
    #command = f'onmt_translate -model {model} -src {src_path} -output {output_path}{args.step} -gpu {args.gpu}'

    src_path = f'{data_path}/encoded/{args.tokenizer}.{args.vocab_size}.{args.set_type}.{src}'

    output_path = f'{model_path}/predictions/encoded/{args.run}.{args.corpus}.{args.set_type}.'
    encoded_output = f'{xmodel}/predictions/encoded/{args.run}.{args.corpus}.{args.set_type}.'
    decoded_output =  f'{xmodel}/predictions/decoded/{args.run}.{args.corpus}.{args.set_type}.'


    dec = """python tokenizer.py --input_path {} --model_directory tokenizers \
            --model_output {} --language {} --vocab_size {} \
            --mode decode --model_type {}"""


    #set onmt for best performing models
    best_blue_model_test = command.format(model_bleu,src_path,output_path+'bleu')
    best_chrf_model_test = command.format(model_chrf,src_path,output_path+'chrf')

    #path to test set
    test_references = test_path+f'.{tgt}'
    test_refs = [open(test_references,'r').readlines()]

    if args.debug == 'True':

        print(best_blue_model_test)
        print(best_chrf_model_test)


    else:
        decoded_predictions =  f'{model_path}/predictions/decoded/{args.run}.{args.corpus}.test.'
        #translate using best blue-evaluated model
        if not os.path.exists(output_path+'bleu'):
            os.system(best_blue_model_test)
        #decode best blue-evaluated model translation
        os.system(dec.format(encoded_output+'bleu',decoded_output+'bleu',args.target, args.vocab_size,args.tokenizer))
        #translate using best chrf-evaluated model
        if not os.path.exists(output_path+'chrf'):
            os.system(best_chrf_model_test)
        #decode best chrf-evaluated model translation
        os.system(dec.format(encoded_output+'chrf',decoded_output+'chrf',args.target, args.vocab_size,args.tokenizer))


        best_bleu_hypothesis = f"{decoded_output}bleu"
        bleu_sys = open(best_bleu_hypothesis,'r').readlines()
        best_bleu_model_bl = bleu.corpus_score(bleu_sys, test_refs).score
        best_bleu_model_cr = chrf.corpus_score(bleu_sys, test_refs).score

        best_chrf_hypothesis = f"{decoded_output}chrf"
        chrf_sys = open(best_chrf_hypothesis,'r').readlines()
        best_chrf_model_bl = bleu.corpus_score(chrf_sys, test_refs).score
        best_chrf_model_cr = chrf.corpus_score(chrf_sys, test_refs).score

        if best_chrf_model != best_bleu_model:
            print(f'Best model evaluated on bleu is at {str(best_bleu_model+1) '000'} iterations')
            print(f'BLEU: {best_bleu_model_bl}')
            print(f'CHRF: {best_bleu_model_cr}')
            print()

            print(f'Best model evaluated on chrf is at {str(best_chrf_model+1) '000'} iterations')
            print(f'BLEU: {best_chrf_model_bl}')
            print(f'CHRF: {best_chrf_model_cr}')
            print()

        else:
            print(f'Best model evaluated on bleu and chrf is at {str(best_chrf_model+1) '000'} iterations')
            print(f'BLEU: {best_chrf_model_bl}')
            print(f'CHRF: {best_chrf_model_cr}')
            print()
    print()
    print(bleu.get_signature())
    print(chrf.get_signature())
