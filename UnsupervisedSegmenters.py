from collections import Counter 
import morfessor
import pickle

class MorfessorTokenizer():
    def __init__(self):
        self.morf_model = morfessor.BaselineModel()
        self.model_path = ''
        self.compounds = []
        self.vocab = []
        self.atoms = []
        
    def prepare_data(self, data, weights):
        words = Counter(open(data, 'r').read().split()).most_common()
        if weights:
            train_data = [(j,i) for i,j in words]
        else:
            train_data = [(1,i) for i,j in words]
        self.compounds = [i for i,j in words]
        self.atoms = Counter("".join(self.compounds)).most_common()
        return train_data
        
        
    def train_model(self,   data, path, 
                    weights = False, 
                    algorithm = 'recursive', 
                   epochs = None,
                    finish_threshold=0.005):
        self.model_path = path
        train_data = self.prepare_data(data, weights) 
        self.morf_model.load_data(train_data)
        self.morf_model.train_batch(algorithm=algorithm,
                                    max_epochs=epochs)
        
        
    #save model when trained
    def save_model (self):
        
        vocabulary = [ "▁"+" ".join(self.morf_model.segment(i)) for i in self.compounds] 
        self.vocab = Counter(" ".join(vocabulary).split()).most_common()  + [(i,1) for i,j in self.atoms+[("▁",1)] if i not in self.vocab] 
        
        with open( self.model_path +'.vocab', 'w') as f:
            for i,j in self.vocab:
                f.write(f"{i}\t{j}\n")
        
        with open( self.model_path +'.pickle', 'wb') as handle:
            model = self.morf_model
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    #load model if availble 
    def load_model (self,path):
        self.model_path = path
        with open(self.model_path, 'rb') as handle:
            self.morf_model = pickle.load(handle)
        try:
            with open(self.model_path.replace('pickle','vocab'), 'r') as f:
                self.vocab = [i.split() for i in f.readlines()]
        except:
            vocabulary = [ "▁"+" ".join(self.morf_model.segment(i)) for i in self.morf_model.get_compounds()]
            self.vocab = Counter(" ".join(vocabulary).split()).most_common() + \
            [(i,1) for i,j in self.atoms+[("▁",1)] if i not in self.vocab] 
        
        
    def segment_word(self, word, n, addcount =1):
        segments = self.morf_model.viterbi_nbest(word,n,addcount )
        if len(segments) < n:
            s = len(segments)-1
        else:
            s = n-1
        return segments[s][0]
    
    
    # returns a list with segmentaions 
    def encode_morf(self, sentence, n = 1 ):
        encode = lambda x: "▁"+" ".join(self.segment_word( x, n))
        if type(sentence) == str:
            encoded_sentence = [encode(i) for i in sentence.split()]
        if type(sentence) == list:
            encoded_sentence = [encode(i)  for i in sentence]
        return encoded_sentence
        
        
    #returns a strings
    def decode_morf(self, sentence):
        decode = lambda x: x.replace(' ','').replace("▁",' ')
        if type(sentence) == str:
            decoded_sentence = decode(sentence)
        if type(sentence) == list:
            decoded_sentence =  decode(" ".join(sentence))
        if len(decoded_sentence ) > 1:
            decoded_sentence = decoded_sentence[1:]
            
        return decoded_sentence 
    
    def encode_file_morf(self, source, path, name, n=1):
        data = open(source, 'r').readlines()
        encode_sent = lambda x: " ".join(self.encode_morf(x, n = 1))
        encode_data = lambda text: ("\n".join([encode_sent(i.split()) for i in text]))
        with open("{}/{}".format(path, name),'w') as f:
            f.write(encode_data(data))
        return 
    
    def decode_file_morf(self, source, path, name):
        data = open(source, 'r').readlines()
        decode_sent = lambda x: " ".join(self.decode_morf(x))
        decode_data = lambda text: ("\n".join([decode_sent(i.split()) for i in text]))
        with open("{}/{}".format(path, name),'w') as f:
            f.write(decode_data(data))
        return 
       
       
  
       
            
    
if __name__ == '__main__':
    foo = "tokenize!"
    
    print(foo)

