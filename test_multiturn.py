from trl.trainer.ppo_config import PPOConfig
from trl.trainer.ppo_trainer import PPOTrainer
### Custom implementations above this line ###

from trl import AutoModelForCausalLMWithValueHead

import torch
from tqdm import tqdm
import pandas as pd

tqdm.pandas()

from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from trl.core import LengthSampler

def main():
    config = PPOConfig(
        model_name="JackFram/llama-160m",
        learning_rate=1.41e-5,
        multiturn_mode = True) # multiturn code change #1
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 1} #changed batch size from 16 to 1, nvm this doesnt do anything lol

    def build_dataset(config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
        """
        Build dataset for training. This builds the dataset from `load_dataset`, one should
        customize this function to train the model on its own dataset.

        Args:
            dataset_name (`str`):
                The name of the dataset to be loaded.

        Returns:
            dataloader (`torch.utils.data.DataLoader`):
                The dataloader for the dataset.
        """
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        # load imdb with datasets
        ds = load_dataset(dataset_name, split="train")
        ds = ds.rename_columns({"text": "review"})
        ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

        input_size = LengthSampler(input_min_text_length, input_max_text_length)

        def tokenize(sample):
            sample["input_ids"] = tokenizer.encode('the' + sample["review"])[: input_size()]
            sample["query"] = tokenizer.decode(sample["input_ids"])
            return sample

        ds = ds.map(tokenize, batched=False)
        ds.set_format(type="torch")
        return ds

    dataset = build_dataset(config)


    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side = 'left')


    tokenizer.pad_token = tokenizer.eos_token

    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)
    ppo_trainer.init_multiturn_mode(terminal_states=["<STOP>"]) # multiturn code change #2

    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)

    output_min_length = 4
    output_max_length = 16
    output_length_sampler = LengthSampler(output_min_length, output_max_length)


    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        #"max_length": 100
    }

    #### test loss mask 
    #test_sequence = "end here this is a test start here and hi hi magikarp end here hello test start here hello hello"
    #encoded_test_sequence = tokenizer.encode(test_sequence, return_tensors='pt').to(device)
    #result = ppo_trainer.custom_mask(encoded_test_sequence, ["start here"], ["end here"])

    #assert False
    ####

    for batch in tqdm(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"] # size is 128
        print(len(query_tensors))

        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            #print(query)
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]


        #### Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

         #### Run PPO step
        print(response_tensors)
        print(tokenizer.decode(response_tensors[0]))
        masked = ppo_trainer.batched_custom_mask(response_tensors, ['the'], ['end'])
        print(masked)

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
    





if __name__ == "__main__":
    main()