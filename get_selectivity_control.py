# Code with some modifications from https://github.com/yanaiela/amnesic_probing/tree/a96e3067a9f9918099015a7173c94830584bbe61
import gc
import time
from evaluation.SelectivityControl import SelectivityControl
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np

np.random.seed(10)


def get_lm_vals(model_name):
    lm_model = BertForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    out_embed = lm_model.cls.predictions.decoder.weight.detach().cpu().numpy()
    bias = lm_model.cls.predictions.decoder.bias.detach().cpu().numpy()
    return lm_model, tokenizer, out_embed, bias


def run_selectivity_control(task: str, method_type: str, method_folder_path: str, original_data_path: str,
                            bert_tokenizer: str):
    lm_model, tokenizer, out_embed, bias = get_lm_vals(bert_tokenizer)

    start = time.time()
    # Run selectivity control
    selectivity_obj = SelectivityControl(tokenizer, out_embed, bias)
    selectivity_obj.run_selectivity_control(input_dir=original_data_path, projection=method_folder_path + '/P.npy',
                                            rebias_labels_name=task,
                                            out_dir=method_folder_path)
    end = time.time()
    print("The time it takes to run selectivity control equal: ",
          (end - start) / 60, " minutes")

    del selectivity_obj
    del tokenizer
    del out_embed
    del bias
    gc.collect()
    return


if __name__ == "__main__":
    ########## Universal Dependency dataset ##############

    ########### MASKED ###################
    run_selectivity_control("dep", "inlp",
                            "results/100k_batches_SGD_stable/masked/dep/removed_inlp", "datasets/ud_data_masked",
                            'bert-base-uncased')
    run_selectivity_control("dep", "mp", "results/100k_batches_SGD_stable/masked/dep/removed_mp",
                            "datasets/ud_data_masked",
                            'bert-base-uncased')

    # pos.pickle -> f-pos
    run_selectivity_control("pos", "inlp", "results/100k_batches_SGD_stable/masked/fpos/removed_inlp",
                            "datasets/ud_data_masked", 'bert-base-uncased')
    run_selectivity_control("pos", "mp", "results/100k_batches_SGD_stable/masked/fpos/removed_mp",
                            "datasets/ud_data_masked",
                            'bert-base-uncased')

    # tag.pickle -> c-pos
    run_selectivity_control("tag", "inlp",
                            "results/100k_batches_SGD_stable/masked/cpos/removed_inlp", "datasets/ud_data_masked",
                            'bert-base-uncased')
    run_selectivity_control("tag", "mp", "results/100k_batches_SGD_stable/masked/cpos/removed_mp",
                            "datasets/ud_data_masked",
                            'bert-base-uncased')

    ######### NORMAL NON-MASKED ###############

    run_selectivity_control("dep", "inlp",
                            "results/100k_batches_SGD_stable/normal/dep/removed_inlp", "datasets/ud_data_normal",
                            'bert-base-uncased')
    run_selectivity_control("dep", "mp", "results/100k_batches_SGD_stable/normal/dep/removed_mp",
                            "datasets/ud_data_normal",
                            'bert-base-uncased')

    # pos.pickle -> f-pos
    run_selectivity_control("pos", "inlp", "results/100k_batches_SGD_stable/normal/fpos/removed_inlp",
                            "datasets/ud_data_normal", 'bert-base-uncased')
    run_selectivity_control("pos", "mp", "results/100k_batches_SGD_stable/normal/fpos/removed_mp",
                            "datasets/ud_data_normal",
                            'bert-base-uncased')

    # tag.pickle -> c-pos
    run_selectivity_control("tag", "inlp",
                            "results/100k_batches_SGD_stable/normal/cpos/removed_inlp", "datasets/ud_data_normal",
                            'bert-base-uncased')
    run_selectivity_control("tag", "mp", "results/100k_batches_SGD_stable/normal/cpos/removed_mp",
                            "datasets/ud_data_normal",
                            'bert-base-uncased')
