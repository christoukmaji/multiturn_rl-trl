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
    found_index_of_end_state = None

    encoded_end_state = test_suit.tokenizer.encode("end here", return_tensors='pt').to(test_suit.device)[:,1:] # so we dnt get start token
    for i in range(encoded_test_sequence.size(1) - encoded_end_state.size(1) + 1):
        if torch.equal(encoded_test_sequence[:, i:i + encoded_end_state.size(1)], encoded_end_state):
            found_index_of_end_state = i
    assert found_index_of_end_state, f"❌ Failed Loss Mask Test: #2: The end state was not correctly encoded by the tokenizer"
    logger.info("✅ Passed Loss Mask Test: #2")



    # Test 3: check that the entire mask is ones, with the exception of the start token
    test_sequence = "start here and hi hi end here"
    encoded_test_sequence = test_suit.tokenizer.encode(test_sequence, return_tensors='pt').to(test_suit.device)
    result = test_suit.ppo_trainer.custom_mask(encoded_test_sequence, ["start here"], ["end here"])

    assert result.size(dim=0) - 1 == result.sum(), f"❌ Failed Loss Mask Test: #3: Tensor is {result} but should have been all 1's, with the exception of the start token at index=0"
    logger.info("✅ Passed Loss Mask Test: #3")


    # Test 4: check that the entire mask is zeros until the start state
    test_sequence = "end here and hi hi"
    encoded_test_sequence = test_suit.tokenizer.encode(test_sequence, return_tensors='pt').to(test_suit.device)
    result = test_suit.ppo_trainer.custom_mask(encoded_test_sequence, ["start here"], ["end here"])

    assert torch.equal(result, torch.zeros(result.shape)), f"❌ Failed Loss Mask Test: #4: Tensor is {result} but should have been all 0's"
    logger.info("✅ Passed Loss Mask Test: #4")


    # Test 5: "Regular" case where we have some tokens, start state, more tokens, end state, some tokens
    test_sequence = "nlp is cool start here and hi hi end here nlp is the best"
    encoded_test_sequence = test_suit.tokenizer.encode(test_sequence, return_tensors='pt').to(test_suit.device)
    result = test_suit.ppo_trainer.custom_mask(encoded_test_sequence, ["start here"], ["end here"])

    # find start state index
    found_index_of_start_state = None
    encoded_start_state = test_suit.tokenizer.encode("start here", return_tensors='pt').to(test_suit.device)[:,1:] # so we dnt get start token
    for i in range(encoded_test_sequence.size(1) - encoded_start_state.size(1) + 1):
        if torch.equal(encoded_test_sequence[:, i:i + encoded_start_state.size(1)], encoded_start_state):
            found_index_of_start_state = i
    
    # find end state index 
    # slight change from other end state test bc here we want the final index of end state
    found_index_of_end_state = None
    encoded_end_state = test_suit.tokenizer.encode("end here", return_tensors='pt').to(test_suit.device)[:,1:] # so we dnt get start token
    for i in range(encoded_test_sequence.size(1) - encoded_end_state.size(1) + 1):
        if torch.equal(encoded_test_sequence[:, i:i + encoded_end_state.size(1)], encoded_end_state):
            found_index_of_end_state = i + encoded_end_state.size(1)
    
    emulate_output = torch.zeros(encoded_test_sequence.shape)
    emulate_output[0, found_index_of_start_state:found_index_of_end_state] = torch.ones(found_index_of_end_state - found_index_of_start_state)
    #print(emulate_output, result)
    #print(encoded_test_sequence)
    #print(encoded_start_state)
    #print(encoded_end_state)
    emulate_output = emulate_output.reshape(-1)

    assert torch.equal(emulate_output, result), f"❌ Failed Loss Mask Test: #5: You are not masking the correct tokens"
    logger.info("✅ Passed Loss Mask Test: #5 - Case where both start and end states are in the middle of the sequence")



    # Test 6:
    # Start state followed by another start state
    test_sequence = "start here nlp is cool start here and hi hi nlp is the best"
    encoded_test_sequence = test_suit.tokenizer.encode(test_sequence, return_tensors='pt').to(test_suit.device)
    result = test_suit.ppo_trainer.custom_mask(encoded_test_sequence, ["start here"], ["end here"])

    assert result.size(dim=0) - 1 == result.sum(), f"❌ Failed Loss Mask Test: #6: Tensor is {result} but should have been all 1's" # with exception of start token
    logger.info("✅ Passed Loss Mask Test: #6 - Case with two consecutive start states")



    # Test 6:
    # End state followed by another end state
    test_sequence = "here nlp is cool end here and hi hi nlp is end here the best"
    encoded_test_sequence = test_suit.tokenizer.encode(test_sequence, return_tensors='pt').to(test_suit.device)
    result = test_suit.ppo_trainer.custom_mask(encoded_test_sequence, ["start here"], ["end here"])

    assert torch.equal(result, torch.zeros(result.shape)), f"❌ Failed Loss Mask Test: #7: Tensor is {result} but should have been all 0's" # with exception of start token
    logger.info("✅ Passed Loss Mask Test: #7 - Case with two consecutive end states")


if __name__ == "__main__":
    all_tests = [test_loss_mask]
    test_object = TestSuite()
    for test in all_tests:
        test(test_object)
    logger.info("✅✅✅ ALL TESTS PASSED ✅✅✅")