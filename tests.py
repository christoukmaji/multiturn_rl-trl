from trl.trainer.ppo_config import PPOConfig
from trl.trainer.ppo_trainer import PPOTrainer

from trl import AutoModelForCausalLMWithValueHead

import torch
from tqdm import tqdm
import pandas as pd
import logging


tqdm.pandas()

from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from trl.core import LengthSampler
from trl.utils.testing_utils import build_dataset, collator

# Set up logging
logging.basicConfig(filename='example.log', level=logging.INFO, 
                    format='%(message)s')
logger = logging.getLogger(__name__)

# Adding a console handler for debugging purposes
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(console_handler)

class TestSuite():
    def __init__(self):
        config = PPOConfig(
            model_name="JackFram/llama-160m",
            learning_rate=1.41e-5,
            multiturn_mode = True) # multiturn code change #1

        config = PPOConfig(
                model_name="JackFram/llama-160m",
                learning_rate=1.41e-5,
                multiturn_mode = True) # multiturn code change #1
        

        dataset = build_dataset(config)

        
        model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side = 'left')


        tokenizer.pad_token = tokenizer.eos_token

        ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)
        ppo_trainer.init_multiturn_mode(terminal_states=["<STOP>"]) # multiturn code change #2

        
        device = ppo_trainer.accelerator.device
        if ppo_trainer.accelerator.num_processes == 1:
            device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

        self.tokenizer = tokenizer
        self.ppo_trainer = ppo_trainer
        self.device = device

def test_loss_mask(test_suit:TestSuite):
    # Test 1: Check that the start state is correctly being encoded into the sequence
    test_sequence = "nlp is cool start here and hi hi end here nlp is the best"
    encoded_test_sequence = test_suit.tokenizer.encode(test_sequence, return_tensors='pt').to(test_suit.device)
    result = test_suit.ppo_trainer.custom_mask(encoded_test_sequence, ["start here"], ["end here"])
    found_index_of_start_state = None

    encoded_start_state = test_suit.tokenizer.encode("start here", return_tensors='pt').to(test_suit.device)[:,1:] # so we dnt get start token
    for i in range(encoded_test_sequence.size(1) - encoded_start_state.size(1) + 1):
        if torch.equal(encoded_test_sequence[:, i:i + encoded_start_state.size(1)], encoded_start_state):
            found_index_of_start_state = i
    assert found_index_of_start_state, f"❌ Failed Loss Mask Test: #1: The start state was not correctly encoded by the tokenizer"
    logger.info("✅ Passed Loss Mask Test: #1")



    # Test 2: Check that the end state is correctly being encoded into the sequence
    test_sequence = "nlp is cool start here and hi hi end here nlp is the best"
    encoded_test_sequence = test_suit.tokenizer.encode(test_sequence, return_tensors='pt').to(test_suit.device)
    result = test_suit.ppo_trainer.custom_mask(encoded_test_sequence, ["start here"], ["end here"])
    found_index_of_start_state = None

    encoded_end_state = test_suit.tokenizer.encode("start here", return_tensors='pt').to(test_suit.device)[:,1:] # so we dnt get start token
    for i in range(encoded_test_sequence.size(1) - encoded_end_state.size(1) + 1):
        if torch.equal(encoded_test_sequence[:, i:i + encoded_end_state.size(1)], encoded_end_state):
            found_index_of_start_state = i
    assert found_index_of_start_state, f"❌ Failed Loss Mask Test: #2: The end state was not correctly encoded by the tokenizer"
    logger.info("✅ Passed Loss Mask Test: #2")



    # Test 3: check that the entire mask is ones, with the exception of the start token
    test_sequence = "start here and hi hi end here"
    encoded_test_sequence = test_suit.tokenizer.encode(test_sequence, return_tensors='pt').to(test_suit.device)
    result = test_suit.ppo_trainer.custom_mask(encoded_test_sequence, ["start here"], ["end here"])

    assert result.size(dim=0) - 1 == result.sum(), f"❌ Failed Loss Mask Test: #3: Tensor is {result} but should have been all 1's, with the exception of the start token at index=0"
    logger.info("✅ Passed Loss Mask Test: #3")


    # Test 4: check that the entire mask is zeros
    test_sequence = "end here and hi hi start here"
    encoded_test_sequence = test_suit.tokenizer.encode(test_sequence, return_tensors='pt').to(test_suit.device)
    result = test_suit.ppo_trainer.custom_mask(encoded_test_sequence, ["start here"], ["end here"])

    assert result.sum() == 0, f"❌ Failed Loss Mask Test: #4: Tensor is {result} but should have been all 0's"
    logger.info("✅ Passed Loss Mask Test: #4")


    # Test 5: "Regular" case where we have some tokens, start state, more tokens, end state, some tokens
    test_sequence = "nlp is cool start here and hi hi end here nlp is the best"
    encoded_test_sequence = test_suit.tokenizer.encode(test_sequence, return_tensors='pt').to(test_suit.device)
    result = test_suit.ppo_trainer.custom_mask(encoded_test_sequence, ["start here"], ["end here"])

    # find start state index
    encoded_start_state = test_suit.tokenizer.encode("start here", return_tensors='pt').to(test_suit.device)

    assert result.sum() == 0, f"❌ Failed Loss Mask Test: #5: You are not masking the correct tokens"
    logger.info("✅ Passed Loss Mask Test: #5")


if __name__ == "__main__":
    all_tests = [test_loss_mask]
    test_object = TestSuite()
    for test in all_tests:
        test(test_object)
    logger.info("✅✅✅ ALL TESTS PASSED ✅✅✅")