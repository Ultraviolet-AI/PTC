
def save_to_pickle(dataset, fileloc):
    """
    Saves dataset to a pickle to specified location
    """
    filename = fileloc + ".pickle"
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)
        logger.info("Saved to {}".format(filename))

def load_from_pickle(fileloc):
    """
    Loads dataset from a pickle
    """
    filename = fileloc + ".pickle"
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    logger.info("Loaded from {}".format(filename))
    return dataset
  
def detokenize(tokenized, tokenizer):
    """
    Tries to detokenize tokenized sequence
    """
    # For each tokens, do:
    for i in range(len(tokenized)):
        # If the token is one of the following, delete space
        if  tokenized[i] in "”’)]}>.,?!:;-_":
            tokenized[i] = "##" + tokenized[i]

        # If current token is not the last, and
        if i != len(tokenized) - 1:
          # If current token is one of the following, delete space in following token
            if tokenized[i] in "“([{<-_" or (tokenized[i] == "’" and tokenized[i + 1] == "s"):
                tokenized[i + 1] = "##" + tokenized[i + 1]
  
    # Revert into string
    reverted = tokenizer.convert_tokens_to_string(tokenized)
    return reverted


def merge_overlapping(indices_list):
    """
    Merges overlapping indices and sorts indices from list of tuples
    """
    # If no propaganda, return empty list
    if indices_list == []:
        return []

    # Sort the list
    indices_list = sorted(indices_list)
    i = 0

    # Going through tuples from the beginning, see if it overlaps with the nex one 
    # and merge it if so
    while True:
        if i == len(indices_list) - 1:
            break
    
    # If the next one is within the range of current, just delete next one. If
    # overlapping, then merge the range
    elif indices_list[i][1] >= indices_list[i + 1][0]:
      if indices_list[i][1] >= indices_list[i + 1][1]:
        pass
      else:
        indices_list[i] = (indices_list[i][0], indices_list[i+1][1])
      indices_list.pop(i+1)

    # Go to next element if not overlapping
    else:
      i += 1

  return indices_list

def article_to_sequences(article_id, article, tokenizer):
  """
  Divides article into sequences, dividing first by sentences then to powersets
  of the sentences
  """
  # Split the lines by sentences
  curr = 0
  lines = article.split("\n")
  sequences = []
  seq_starts = []
  seq_ends = []

  # For each lines, do:
  for line in lines:
    # If an empty line, just continue
    if line == "":
      curr += 1
      continue

    # Tokenize the line
    tokenized = tokenizer.tokenize(line)

    # For each token, do:
    seq_start = 0
    for ind, token in enumerate(tokenized):
      # Get the token without ## sign
      mod_start_token = token.replace("##", "")

      # Find the start of the sequence in line
      seq_start = line.lower().find(mod_start_token, seq_start)

      # Update the end of the sequence
      seq_end = seq_start

      # For each following tokens in the line, do
      for iter in range(1, len(tokenized) + 1 - ind):
        # Also modify this token
        mod_end_token = tokenized[ind + iter - 1].replace("##", "")
        # Find the end of the token
        seq_end = line.lower().find(mod_end_token, seq_end) + len(mod_end_token)

        sequences.append(tokenizer.convert_tokens_to_string(tokenized[ind: ind + iter]))
        seq_starts.append(curr + seq_start)
        seq_ends.append(curr + seq_end)

      # Update the start of the sequence
      seq_start += len(mod_start_token)

    # Update the current whereabouts
    curr += len(line) + 1

  dataframe = pandas.DataFrame(None, range(len(sequences)), ["id", "seq_starts", "seq_ends", "label", "text"])
  dataframe["id"] = [article_id] * len(sequences)
  dataframe["seq_starts"] = seq_starts
  dataframe["seq_ends"] = seq_ends
  dataframe["label"] = [0] * len(sequences)
  dataframe["text"] = sequences
  return dataframe

def article_labels_to_sequences(article, indices_list):
  """
  Divides article into sequences, where each are tagged to be propaganda or not
  """
  # Start at 0 indices, and split the article into lines
  curr = 0
  lines = article.split("\n")
  sequences = {}

  # For each lines, do:
  for line in lines:
    # If an empty line, just continue after adding \n character
    if line == "":
      curr += 1
      continue

    # If nothing in indices_list or current line is not part of propaganda, 
    # just mark it to be none 
    elif indices_list == [] or curr + len(line) <= indices_list[0][0]:
      sequences[line] = 0

    # If current line is part of propaganda, do:
    else:
      # If the propaganda is contained within the line, add it accordingly
      # and pop that indices range
      if curr + len(line) >= indices_list[0][1]:
        sequences[line[:indices_list[0][0] - curr]] = 0
        sequences[line[indices_list[0][0] - curr:indices_list[0][1] - curr]] = 1
        sequences[line[indices_list[0][1] - curr:]] = 0
        indices_list.pop(0)
      # If the propaganda goes over to the next line, add accordingly and 
      # modify that indices range
      else:
        sequences[line[:indices_list[0][0] - curr]] = 0
        sequences[line[indices_list[0][0] - curr:]] = 1
        indices_list[0][0] = curr + len(line) + 2

    # Add the current line length plus \n character
    curr += len(line) + 1

  dataframe = pandas.DataFrame(None, range(len(sequences)), ["label", "text"])
  dataframe["label"] = sequences.values()
  dataframe["text"] = sequences.keys()
  return dataframe

def articles_to_dataframe(article_folder, label_folder, task="SI"):
  """
  Preprocesses the articles into dataframes with sequences with binary tags
  """
  # First sort the filenames and make sure we have label file for each articles
  article_filenames = sorted(glob.glob(os.path.join(article_folder, "*.txt")))
  label_filenames = sorted(glob.glob(os.path.join(label_folder, "*.labels")))
  assert len(article_filenames) == len(label_filenames)

  # Initialize sequences
  sequences = []

  # For each article, do:
  for i in range(len(article_filenames)):
    # Get the id name
    article_id = os.path.basename(article_filenames[i]).split(".")[0][7:]

    # Read in the article
    with codecs.open(article_filenames[i], "r", encoding="utf8") as f:
      article = f.read()

    # Read in the label file and store indices for SI task
    if task == "SI":
      with open(label_filenames[i], "r") as f:
        reader = csv.reader(f, delimiter="\t")
        indices_list = []
        for row in reader:
          indices_list.append([int(row[1]), int(row[2])])

        # Merge the indices if overlapping
        indices_list = merge_overlapping(indices_list)

      # Add to the sequences
      sequences.append(article_labels_to_sequences(article, indices_list))

    # Read in the label file and store indices for TC task
    elif task == "TC":
      with open(label_filenames[i], "r") as f:
        reader = csv.reader(f, delimiter="\t")
        article_sequences = []
        labels_list = []
        for row in reader:
          article_sequences.append(article[int(row[2]):int(row[3])])
          labels_list.append(PROP_TECH_TO_LABEL[row[1]])

      sequence = pandas.DataFrame(None, range(len(article_sequences)), ["label", "text"])
      sequence["label"] = labels_list
      sequence["text"] = article_sequences

      # Add to the sequences
      sequences.append(sequence)

    else:
      logger.error("Undefined task %s !", task)

  # Concatenate all dataframes
  dataframe = pandas.concat(sequences, ignore_index=True)

  return dataframe

def convert_dataframe_to_features(dataframe, max_seq_length, tokenizer):
  """
  Converts dataframe into features dataframe, where each feature will
  take form of [CLS] + A + [SEP]
  """
  # Create features
  features = pandas.DataFrame(None, range(dataframe.shape[0]), 
                              ["input_ids", "input_mask", "segment_ids", "label_ids"])

  # For each sequence, do:
  for i in range(len(dataframe)):
    # Set first and second part of the sequences
    tokens = tokenizer.tokenize(dataframe["text"][i])

    # If length of the sequence is greater than max sequence length, truncate it
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]

    # Concatenate the tokens
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]

    # Compute the ids
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    input_ids = input_ids + [pad_token] * padding_length
    input_mask = input_mask + [0] * padding_length
    segment_ids = segment_ids + [0] * padding_length
    label_id = dataframe["label"][i]

    # Assert to make sure we have same length
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    # Put the data into features dataframe
    features["input_ids"][i] = input_ids
    features["input_mask"][i] = input_mask
    features["segment_ids"][i] = segment_ids
    features["label_ids"][i] = label_id

  return features

def generate_training_dataset_from_articles(articles_folders, labels_folders, tokenizer, task="SI"):
  """
  Generates dataset to go into BERT from articles and labels
  """
  # If generating dataset for evaluation, do:
  logger.info("Generating training dataset...")
    
  # For each articles and labels folder set, turn them into dataframes
  dataframe_list = []
  for i in range(len(articles_folders)):
    logger.info("Generating dataframe for folder %s", articles_folders[i])
    dataframe_list.append(articles_to_dataframe(articles_folders[i], labels_folders[i], task=task))

  # Concatenate the dataframes to make a total dataframe
  dataframe = pandas.concat(dataframe_list, ignore_index=True)

  print(dataframe)
  print(dataframe.shape)

  # Process into features dataframe
  logger.info("Creating features from dataframe")
  features = convert_dataframe_to_features(dataframe, args['max_seq_length'], tokenizer)
      
  # Creating TensorDataset from features
  logger.info("Creating TensorDataset from features dataframe")
  all_input_ids = torch.tensor(features["input_ids"], dtype=torch.long)
  all_input_mask = torch.tensor(features["input_mask"], dtype=torch.long)
  all_segment_ids = torch.tensor(features["segment_ids"], dtype=torch.long)
  all_label_ids = torch.tensor(features["label_ids"], dtype=torch.long)

  dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
  return dataset
