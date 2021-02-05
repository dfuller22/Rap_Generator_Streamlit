class TextGenerator():
    
    def __init__(self, model, char_ref, idx_ref):
        self.__model = model
        self.__char_ref = char_ref
        self.__idx_ref = idx_ref
        
    def generate_text(self, start_string, num_chars=100, temp=1.0):
        # https://www.tensorflow.org/tutorials/text/text_generation#the_prediction_loop
        import tensorflow as tf

        ## Low temperature results in more predictable text.
        ## Higher temperature results in more surprising text.
        temperature = temp

        ## Number of characters to generate
        num_generate = num_chars

        ## Converting our start string to numbers (vectorizing)
        input_eval = [self.__idx_ref[s] for s in start_string.lower()]
        input_eval = tf.expand_dims(input_eval, 0)

        ## Empty string to store our results
        text_generated = []

        ## Here batch size == 1
        self.__model.reset_states()

        for i in range(num_generate):
            predictions = self.__model(input_eval)
            ## Remove the batch dimension
            predictions = tf.squeeze(predictions, 0)

            ## Using a categorical distribution to predict the character returned by the model
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

            ## Pass the predicted character as the next input to the model
            ## along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(self.__char_ref[predicted_id])

        self._generated_text = (start_string + ''.join(text_generated))
        
    def get_generated_text(self):
        return self._generated_text

class Censor():
    
    def __init__(self, lyrics):
        self._lyrics = lyrics
        self._targeted = None
        self._token_set = None
        self.__censored_words = ['ass', 'asses', 'fuck', 'fucked', 'fucker', 'fucks', 'fuckin',
                        'fucking', 'motherfuck', 'motherfucker', 'motherfuckin', 'ho', 'hoes',
                        'hoe', 'motherfucking', 'bitch', 'bitches', 'cock', 'dick',
                        'dicks', 'pussy', 'pussies', 'shit', 'shits', 'shitty',
                        'faggot', 'fag', 'fags', 'nigga', 'niggas', 'nigger']
        
    def create_set(self, show=False):
        ## Set creation + stat assignment
        self._token_set = list(set(self._lyrics))
        self.__token_num_total = len(self._lyrics)
        self.__token_num_unique = len(set(self._token_set))
        
        ## Optional Q.C.
        if show:
            print(f'Total tokens: {self.__token_num_total:,}')
            print(f'Total unique tokens: {self.__token_num_unique:,}')
            
    def find_targets(self, extra_targets=None):
        ## Check if additional censors needed
        if isinstance(extra_targets, list):
            self.__censored_words.extend(extra_targets)
        
        ## Check for smaller version of lyrics
        if self._token_set:
            targets_within = []

            ## Collecting only the words applicable to these lyrics
            for word in self.__censored_words:
                if word in self._token_set:
                    targets_within.append(word)
            
            ## Assignment of lyric-specific curses
            self._targeted = targets_within
        
        ## Kick out if not done yet
        else:
            print(10*'--', 'Error', 10*'--')
            return 'Please use .create_set() first..'
            
    def count_targets(self, lyric_type='I', show=False):
        results = {}
        
        ## Check for necessary info
        if self._targeted == None:
            print(10*'--', 'Error', 10*'--')
            return 'Please use .find_targets() first..'
        
        ## Type of lyrics to count
        if lyric_type == 'I':
            lyrics2count = self._lyrics
        elif lyric_type == 'A':
            lyrics2count = self._altered_lyrics
        elif lyric_type == 'C':
            lyrics2count = self._cleaned_lyrics
        elif lyric_type == 'M':
            lyrics2count = self._muted_lyrics
        else:
            return "Please use 'I', 'A', 'C', or 'M' for 'lyric_type' parameter. See docstring for more info."
        
        ## Begin counting
        for word in lyrics2count:
            ## Pull curses and increment
            if word in self._targeted:
                try:
                    results[word] += 1
                except KeyError:
                    results[word] = 1
        
        ## Sort + assign to attribute
        self.__target_counts = {k:v for k,v in sorted(results.items(), key=lambda item:item[1], reverse=True)}
        
        ## Optional display
        if show:
            return self.__target_counts
        
    def alter_targets(self, replacements):
        ## Set copy for manipulation
        self._altered_lyrics = self._lyrics.copy()
        
        ## Check for necessary info + create dict to map values
        try:
            trgt2cnsrd = dict.fromkeys(self._targeted, None)
        except NameError:
            print(10*'--', 'Error', 10*'--')
            return 'Please use .find_targets() first..'
        
        ## Check replacements match dictionary
        assert len(trgt2cnsrd) == len(replacements), "Make sure 'replacement' list is equal in length to targeted words"
        
        ## Matching up dictionary to replacement list
        for i, key in enumerate(trgt2cnsrd):
            trgt2cnsrd[key] = replacements[i]
            
        ## Altering strings
        ## Iter. over each target/replacement
        for key, val in trgt2cnsrd.items():
            ## Iter. over each lyric token
            for i, token in enumerate(self._altered_lyrics):
                ## Regex to sub token inplace + make_copy check
                if key == token:
                    self._altered_lyrics[i] = val
        
    def mute_targets(self, replacement='*'):
        ## Set copy for manipulation
        self._muted_lyrics = self._lyrics.copy()

        ## Iter. over each target word
        for target in self._targeted:
            ## Iter. over each lyric token
            for i, token in enumerate(self._muted_lyrics):
                ## Assigning replacement per target size
                if (target == token) | (target in token):
                    if len(token) == 1:
                        self._muted_lyrics[i] = replacement
                    elif len(token) == 2:
                        self._muted_lyrics[i] = token[0] + replacement
                    elif len(token) == 3:
                        token = token[0] + replacement*2
                        self._muted_lyrics[i] = token
                    elif len(token) >= 4:
                        token = token[0] + replacement*(len(token)-2) + token[-1]
                        self._muted_lyrics[i] = token
                    else:
                        print('Length of <=0!')
        
    def remove_targets(self):
        ## Set copy for manipulation
        self._cleaned_lyrics = self._lyrics.copy()
        
        ## Iter. over each target token
        for target in self._targeted:
            ## Iter. over each lyric token
            for token in self._cleaned_lyrics:
                ## Remove each match to avoid errors
                if (token == target) or (target in token):
                    self._cleaned_lyrics.remove(target)
        
    def add_target(self, to_add, multi_add=False, show=False):
        ## Iter. over each new target OR append .targeted
        if multi_add:
            for new_trgt in to_add:
                if new_trgt in self._targeted:
                    continue
                else:
                    self._targeted.append(new_trgt)
        else:
            if to_add in self._targeted:
                return f"'{to_add}' already a target!"
            else:
                self._targeted.append(to_add)
        
        ## Optional display
        if show:
            return self._targeted

    def str_splitter(self):
        ## Results container
        results = []

        ## Iter. over each character join each non-spacing char.
        for char in self._lyrics:
            ## Spaces indicate where words end
            if (char == ' ') | (char == '\n'):
                results.append(word)
                results.append(char)
                word = '' ## Reset for next word
            else:
                ## Try/Except to handle resets
                try:
                    word = word + char
                except NameError:
                    word = char

        ## Last minute check for words
        if word:
            results.append(word)
            
        self._lyrics = results

    def str_joiner(self, lyric_type='I'):
        ## Type of lyrics to join
        joinable = self.get_lyrics(lyric_type)

        ## Iter. over each token
        for token in joinable:
            ## Try/Except to handle first entry
            try:
                tokens_joined = tokens_joined + token
            except NameError:
                tokens_joined = token

        self._joined_lyrics = tokens_joined

    def execute_censoring(self, add_targ=None):
        ## Create set + find targeted words
        self.create_set()
        self.find_targets()

        ## Check for additional targets to be muted
        if isinstance(add_targ, str):
            self.add_target(add_targ)
        elif isinstance(add_targ, list):
            self.add_target(add_targ, multi_add=True)
        
        ## Execute muting
        self.mute_targets()

    ######################################## GETTERS ########################################

    def get_lyrics(self, lyric_type='I'):
        ## Type of lyrics to return
        if lyric_type == 'I':
            return self._lyrics
        elif lyric_type == 'A':
            return self._altered_lyrics
        elif lyric_type == 'C':
            return self._cleaned_lyrics
        elif lyric_type == 'M':
            return self._muted_lyrics
        elif lyric_type == 'J':
            return self._joined_lyrics
        else:
            return "Please use 'I', 'A', 'C', 'J', or 'M' for 'lyric_type' parameter. See docstring for more info."

    def get_word_counts(self, lyric_type='I'):
        ## Result container
        results = {}
        
        ## Type of lyrics to count
        lyrics2count = self.get_lyrics(lyric_type)

        ## Begin counting
        for word in lyrics2count:
            ## Find words and increment
                try:
                    results[word] += 1
                except KeyError:
                    results[word] = 1
        
        ## Sort + assign to attribute
        word_counts = {k:v for k,v in sorted(results.items(), key=lambda item:item[1], reverse=True)}
        
        ## Display
        return word_counts


def build_app_model(lyrics_string, embed_dim=256, rnn_units=1024):
    import tensorflow as tf

    ## Directory where the checkpoints will be saved
    chkpt_dir = './training_checkpoints_all_songs'
    ## Collection of unique words in corpus
    vocab = sorted(set(lyrics_string))
    ## Length of different characters in dataset
    vocab_size = len(vocab)
    ## Dimenson of word embedding space
    ## NO. RNN units for GRU layer

    ## New model for generation, loading weights + setting tensor shape for input
    model = build_model(vocab_size, embed_dim, rnn_units, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(chkpt_dir))
    model.build(tf.TensorShape([1, None]))

    return model


######################################## SOURCED MATERIAL ########################################

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    import tensorflow as tf

    ## Initial structure provided by:
    ## https://www.tensorflow.org/tutorials/text/text_generation#build_the_model
    
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

