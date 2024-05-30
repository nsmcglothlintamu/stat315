from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def interact_with_llama(prompt):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = AutoTokenizer.from_pretrained('/scratch/user/nsmcglothlin/llama/llama/tokenizer.py')

    # Load pre-trained model
    model = AutoModelForCausalLM.from_pretrained('/scratch/user/nsmcglothlin/llama/llama/model.py')

    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate the sequence
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)

    # Decode the output
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    return decoded_output

def main():
    # Take user input
    user_prompt = input("Enter your prompt for Llama 2: ")
    
    # Generate response
    response = interact_with_llama(user_prompt)

    # Print response
    print(f"Llama 2's response:\n{response}")

if __name__ == "__main__":
    main()

