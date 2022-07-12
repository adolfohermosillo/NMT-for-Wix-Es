import sentencepiece as spm
from UnsupervisedSegmenters import MorfessorTokenizer as morfessor
import argparse
import os

argp = argparse.ArgumentParser()

input_path = argp.add_argument('--input_path',
                               default="",
                           help='Set path to input text for either training, encodind, and decoding')

model_directory = argp.add_argument('--model_directory',
                           help='Set path to where the models will be / is saved')

model_output = argp.add_argument('--model_output',
                                   default="",
                           help='Set path to where model output will be written')

language = argp.add_argument('--language',
                           help='Set the language')


vocab_size = argp.add_argument('--vocab_size',
                            default = "",
                           help='Select ')

model_type = argp.add_argument('--model_type',
                            default = "",
                           help='Select from [bpe, unigram, char, word]')

mode =  argp.add_argument('--mode',
                            default = "train",
                           help='Select from [train, encode, decode]')
args = argp.parse_args()
directory = os.getcwd()

assert args.model_type != "", "You forgot to specify 'model_type'!"
assert args.input_path != "", "You forgot to specify 'input_path'!"
assert args.language != "", "You forgot to specify 'language'!"
assert os.path.exists( f"/{directory}/{args.input_path}"), f"The file {f'{directory}/{args.input_path}'} does not exist!"

if  args.model_type in ['bpe', 'unigram','char']:
    assert args.vocab_size != "", "You forgot to specify 'vocab_size',choose 1000 for char!"


if args.mode == 'train':


    if  args.model_type in ['bpe', 'unigram','char','word']:

        model_prefix=f'{directory}/{args.model_directory}/{args.model_type}/{args.language}.{args.vocab_size}'

        spm.SentencePieceTrainer.train(input=f'{directory}/{args.input_path}',
                                       model_prefix = model_prefix,
                                       vocab_size=int(args.vocab_size),
                                       model_type=args.model_type)

        assert os.path.exists(f'{model_prefix}.model'), ' Model was not saved '
        assert os.path.exists(f'{model_prefix}.vocab'), ' Vocab was not saved '

    if args.model_type == 'morf' :

        morf = morfessor()

        model_prefix= f'{directory}/{args.model_directory}/{args.model_type}/{args.language}'

        morf.train_model(data=f'{directory}/{args.input_path}', path= model_prefix)

        morf.save_model()
        
        
        

else:
    # assert path to file is given
    assert args.model_output != "", "You forgot to specify 'model_output'!"

    # path to model
    model_file =f'{directory}/{args.model_directory}/{args.model_type}/{args.language}'

    if args.model_type in ['bpe', 'unigram','char']:
        model_file = model_file+f'.{args.vocab_size}.model'
        # assert model above exists 
        assert os.path.exists(model_file), f'The path: {model_file} does not exist'
        sp = spm.SentencePieceProcessor(model_file=model_file)
        encode = lambda x: " ".join(sp.encode(x, out_type=str))
        decode = lambda x: sp.decode(x.split())

    if args.model_type == 'morf' :
        model_file = model_file+f'.pickle'

        # assert model above exists 
        assert os.path.exists(model_file), f'The path: {model_file} does not exist'

        morf = morfessor()
        morf.load_model(model_file)
        encode = lambda x:" ".join(morf.encode_morf(x))
        decode = lambda x:" ".join(morf.decode_morf(x).split()).replace('\n','').replace('<unk>','')

    if args.model_type == 'word':
        encode = lambda x: x.replace('\n','')
        decode = lambda x: x.replace('<unk>','').replace('\n','')

    func = {'encode': encode, 'decode':decode }


    with open(f'{directory}/{args.model_output}', 'w') as f:
        data_input = open(f'{directory}/{args.input_path}', 'r').readlines()
        # print(data_input[0])
        output = [func[args.mode](i) for i in data_input]
        #  print(output[0])
        f.write("\n".join(output))

    assert os.path.exists(f'{directory}/{args.model_output}'), f" The file {f'{directory}/{args.model_output}'} was not saved"

