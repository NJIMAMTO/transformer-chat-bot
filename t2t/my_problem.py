from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
 
 
@registry.register_problem
class Translate_JPEN(text_problems.Text2TextProblem):
    @property
    def approx_vocab_size(self):
        return 2**13
    
    @property
    def is_generate_per_split(self):
        return False
 
    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]
 
    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        filename_input = '/content/drive/My Drive/Colab Notebooks/input_corpus.txt'
        filename_output = '/content/drive/My Drive/Colab Notebooks/output_corpus.txt'
 
        with open(filename_input) as f_in, open(filename_output) as f_out:
            for src, tgt in zip(f_in, f_out):
                src = src.strip()
                tgt = tgt.strip()
                if not src or not tgt:
                    continue
                yield {'inputs': src, 'targets': tgt}