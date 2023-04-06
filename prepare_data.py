import json

# Load JSON data from file
with open('output.clean.json', 'r') as f:
    conversations = json.load(f)

# Loop through conversations and write to separate text files
for i, conversation in enumerate(conversations):
    filename = f'data/conversation{i+1}.txt'
    with open(filename, 'w') as f:
        for message in conversation:
            text = message['text']
            user = message['user']
            if user == 'user':
                user = 'prompter'
            else:
                user = 'assistant'
            f.write('<|' + user + '|>' + text + '<|endoftext|>')