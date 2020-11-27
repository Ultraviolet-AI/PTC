
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



# GENERATE SI

def generate_SI_eval_dataset_from_article(article_file, tokenizer):
    """
    Generates SI dataset to go into BERT from articles and labels
    """
    # If generating dataset for evaluation, do:
    logger.info("Generating evaluation dataset...")

    # Get the id name
    article_id = os.path.basename(article_file).split(".")[0][7:]

    # Read in the article
    with codecs.open(article_file, "r", encoding="utf8") as f:
        article = f.read()

  # For evaluating, do:
    dataframe = article_to_sequences(article_id, article, tokenizer)

    # Only include sequences that has alphanumeric characters in them
    bool_indices = [False] * len(dataframe)
    for i in range(len(dataframe)):
        if any(c.isalnum() for c in dataframe["text"][i]):
            bool_indices[i] = True
    dataframe = dataframe.loc[bool_indices].reset_index(drop=True)

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
    return dataset, dataframe



def generate_TC_eval_dataset_from_article(article_folder, indices_file, tokenizer):
    """
    Generates TC dataset to go into BERT from articles and labels
    """
    # If generating dataset for evaluation, do:
    logger.info("Generating evaluation dataset...")

    # First sort the filenames and make sure we have label file for each articles
    article_filenames = sorted(glob.glob(os.path.join(article_folder, "*.txt")))
    articles = {}

    # For each article, read them in:
    for i in range(len(article_filenames)):
        article_id = os.path.basename(article_filenames[i]).split(".")[0][7:]
        with codecs.open(article_filenames[i], "r", encoding="utf8") as f:
            articles[article_id] = f.read()

    # Read in indices file
    with open(indices_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        ids_list = []
        seq_starts = []
        seq_ends = []
        article_sequences = []
        for row in reader:
            ids_list.append(row[0])
            seq_starts.append(row[2])
            seq_ends.append(row[3])
            article_sequences.append(articles[row[0]][int(row[2]):int(row[3])])

    dataframe = pandas.DataFrame(None, range(len(ids_list)), ["id", "seq_starts", "seq_ends", "label", "text"])
    dataframe["id"] = ids_list
    dataframe["seq_starts"] = seq_starts
    dataframe["seq_ends"] = seq_ends
    dataframe["label"] = [0] * len(ids_list)
    dataframe["text"] = article_sequences

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
    return dataset, dataframe



def train(train_dataset, model, tokenizer):
    """
    Trains the model with training dataset
    """
    # Initialize various necessary objects
    tb_writer = SummaryWriter()
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args['train_batch_size'])

    # Compute the total time
    t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']

    # Set the grouped parameters for optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    # Compute warmup steps
    warmup_steps = math.ceil(t_total * args['warmup_ratio'])
    args['warmup_steps'] = warmup_steps if args['warmup_steps'] == 0 else args['warmup_steps']

    # Initialize optimizer as Adam with constant weight decay and a linear scheduler with warmup
    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args['warmup_steps'], num_training_steps=t_total)

    # Initialize variables for training
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args['num_train_epochs']), desc="Epoch")

    # Start training!
    for _ in train_iterator:
        epoch_iterator = tqdm_notebook(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2], 
                    'labels':         batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]

            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']
          
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

            tr_loss += loss.item()
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step() 
                model.zero_grad()
                global_step += 1

                if args['logging_steps'] > 0 and global_step % args['logging_steps'] == 0:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args['logging_steps'], global_step)
                    logging_loss = tr_loss

                if args['save_steps'] > 0 and global_step % args['save_steps'] == 0:
                    output_dir = os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(output_dir)
                logger.info("Saving model checkpoint to %s", output_dir)

    return global_step, tr_loss / global_step



def classify_per_article(eval_dataframe, eval_dataset, model, tokenizer):
    """
    Classifies a single article dataset and returns article id with indices list
    """
    # Load the eval data and initialize sampler
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

    # Start Classification
    preds = None

    # For each batch, evaluate
    for batch in tqdm_notebook(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels':         batch[3]}
            outputs = model(**inputs)
            logits = outputs[1]

        # Get predictions
        if preds is None:
            preds = logits.detach().cpu().numpy()
        else:
            preds = numpy.append(preds, logits.detach().cpu().numpy(), axis=0)

    # Get the most probable prediction
    preds = numpy.argmax(preds, axis=1)
    eval_dataframe["label"] = preds

    # Get the propaganda indices and return them
    eval_dataframe = eval_dataframe.loc[eval_dataframe["label"] == 1].reset_index(drop=True)
    indices_list = []
    for i in range(len(eval_dataframe)):
        indices_list.append(tuple([eval_dataframe["seq_starts"][i], eval_dataframe["seq_ends"][i]]))
    indices_list = merge_overlapping(indices_list)

    return eval_dataframe["id"][0], indices_list



def classify_techniques(eval_dataframe, eval_dataset, model, tokenizer):
    """
    Classifies a single article dataset and returns article id with indices list
    """
    # Load the eval data and initialize sampler
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

    # Start Classification
    logger.info("***** Running classification for article {} *****".format(eval_dataframe["id"][0]))
    logger.info("  Num sequences = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    preds = None

    # For each batch, evaluate
    for batch in tqdm_notebook(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels':         batch[3]}
            outputs = model(**inputs)
            logits = outputs[1]

    # Get predictions
    if preds is None:
        preds = logits.detach().cpu().numpy()
    else:
        preds = numpy.append(preds, logits.detach().cpu().numpy(), axis=0)

    # Get the most probable prediction
    preds = numpy.argmax(preds, axis=1)

    return preds