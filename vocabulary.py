from rdkit import Chem
import re
import numpy as np
import pandas as pd


class Vocabulary:
    def __init__(self, path, max_len=100):

        self.max_len = max_len
        self.path = path

        with open(path, 'r') as f:  # path = 'Vocab.txt'
            vocab = f.read().split()  # list

        # to_add_tokens = ['G', 'A', 'Z', '[Yb]', '[Lu]', '[Ta]', '[W]', '[Re]', '[Os]', '[Pt]', '[Ir]', '[Au]', '[TlH2]', '[PbH]']
        to_add_tokens = ['A', 'G']
        vocab = to_add_tokens + vocab

        # encode
        self.char_to_int = dict()
        for i, char in enumerate(vocab):
            self.char_to_int[char] = i

        # decode
        self.int_to_char = dict()
        for i, char in enumerate(vocab):
            self.int_to_char[i] = char

        self.vocab_size = len(vocab)
        self.name = 1
        print('Number of unique characters in the vocabulary: {} '.format(self.vocab_size))

    def update_vocab(self, smiles):
        '''
        Updates da vocabulary using a list of smiles.
        reads a list of smiles and returns the vocabulary(=all the characters
        in the smiles' list)'''

        # regex = '(\[[^\[\]]{1,6}\])'# finds tokens of the format '[x]'
        regex = '(\[[^\[\]]{1,10}\])'
        unique_chars = set()
        for i, sm in enumerate(smiles):

            # substituir 'Br' por 'R' e 'Cl' por'L'
            sm = sm.replace('Br', 'R').replace('Cl', 'L')
            # print(sm)
            # split each individual smiles string into a list of substrings
            # ['CC1(C)C(=O)NN=C1c1ccc(NC2=C(Cc3cccc([N+](=O)[O-])c3)C(=O)CCC2)cc1F'] becomes:
            # >>['CC1(C)C(=O)NN=C1c1ccc(NC2=C(Cc3cccc(', '[N+]', '(=O)', '[O-]', ')c3)C(=O)CCC2)cc1F']
            sm_chars = re.split(regex, sm)
            for section in sm_chars:
                # print(section)
                if section.startswith('['):  # finds tokens of the format '[x]]
                    unique_chars.add(section)
                else:
                    for char in section:
                        unique_chars.add(char)
        # adding start 'GO', end 'EOS' and padding tokens
        to_add_tokens = ['A', 'G']
        for token in to_add_tokens:
            unique_chars.add(token)

        unique_chars = sorted(unique_chars)
        # Saving to file
        with open(self.path, 'w') as f:
            for char in unique_chars:
                f.write(char + "\n")

        print('Number of unique characters in the vocabulary: {} '.format(len(unique_chars)))

        vocab = sorted(list(unique_chars))
        # encode
        self.char_to_int = dict()
        for i, char in enumerate(vocab):
            self.char_to_int[char] = i

        # decode
        self.int_to_char = dict()
        for i, char in enumerate(vocab):
            self.int_to_char[i] = char

        self.vocab_size = len(vocab)
        # return unique_chars

    def tokenize(self, smiles, pad_end=True):  # Transforms a List of SMILES Strings into a list of tokenized SMILES (where each tokenized SMILES is in itself a list)
        '''
        Parameters
        ----------
        smiles : List of SMILES Strings

        Returns
        -------
        A List of Lists where each entry corresponds to one original SMILES string
         Each SMILES String becomes a List of tokens where Br is replaced by R,
         Cl is replaced by L and sections [x] are a single token. A START,
         PADDING and END token is also added

        '''
        list_tok_smiles = []  # to save each tokenized SMILES String
        for smile in smiles:
            regex = '(\[[^\[\]]{1,10}\])'
            # print(smile)
            smile = smile.replace('Br', 'R').replace('Cl', 'L')
            smile_chars = re.split(regex, smile)
            smile_tok = []
            smile_tok.append('G')  # adding the START token
            for section in smile_chars:
                if section.startswith('['):  # finds tokens of the format '[x]'
                    smile_tok.append(section)
                else:  # section includes many chars
                    for char in section:
                        smile_tok.append(char)

            if pad_end:
                assert len(smile_tok) <= self.max_len
                    

                # padding
                if len(smile_tok) < self.max_len:
                    dif = self.max_len - len(smile_tok)
                    [smile_tok.append('A') for _ in range(dif)]

                # assert len(smile_tok) == self.max_len + 1
                # print(smile_tok)

            list_tok_smiles.append(smile_tok)
        return (list_tok_smiles)

    def encode(self, tok_smiles):
        '''
            Encodes each tokenized SMILES String in 'tok_smiles'

        Parameters
        ----------
        smiles : TYPE List of Lists
            DESCRIPTION. List of tokenized SMILES (List)

        Returns
        -------
        encoded_smiles : TYPE List of List
            DESCRIPTION. List of encoded SMILES (List)

        '''
        encoded_smiles = []
        for smile in tok_smiles:
            smile_idx = []
            for char in smile:
                smile_idx.append(self.char_to_int[char])

            encoded_smiles.append(smile_idx)
        return encoded_smiles

    def decode(self, encoded_smiles):
        '''


        Parameters
        ----------
        encoded_smiles : TYPE
            DESCRIPTION.

        Returns
        -------
        smiles : TYPE List of smiles strings
            DESCRIPTION.

        '''
        smiles = []
        for e_smile in encoded_smiles:
            smile_chars = []
            for idx in e_smile:
                if self.int_to_char[idx] == 'G':
                    continue
                if self.int_to_char[idx] == 'A':
                    break
                smile_chars.append(self.int_to_char[idx])

            smile_str = ''.join(smile_chars)
            smile_str = smile_str.replace('R', 'Br').replace('L', 'Cl')

            smiles.append(smile_str)

        return smiles  # list

    def one_hot_encoder(self, smiles_list, pad_end=True):
        '''


         Parameters
         ----------
         smiles_list : TYPE list of tokenized SMILES
             DESCRIPTION.

         Returns
         -------
         smiles_one_hot : TYPE 3d numpy array with shape (total_number_smiles, max_lenght_sequence, vocab_size) ready to give as input to the a LSTM model
             DESCRIPTION.

         '''

        # smiles_one_hot = []
        print("SMILES", len(smiles_list))
        smiles_one_hot = np.zeros((len(smiles_list), self.max_len if pad_end else len(smiles_list[0]), self.vocab_size), dtype=np.int8)
        for j, smile in enumerate(smiles_list):

            # X = np.zeros((self.max_len, self.vocab_size), dtype=np.int8)
            for i, c in enumerate(smile):
                # print(len(smile), i, c)
                smiles_one_hot[j, i, self.char_to_int[c]] = 1
            # smiles_one_hot.append(X)
            # smiles_one_hot[j] = X
        return smiles_one_hot

    def one_hot_decoder(self, smiles_array):
        '''

        Parameters
        ----------
        smiles_array : TYPE 3d numpy array with shape (total_number_smiles, max_lenght_sequence, vocab_size)
            DESCRIPTION.

        Returns
        -------
        encoded_smiles : TYPE a list of numpy arrays with shape (max_length_sequence,)
            DESCRIPTION.

        '''

        encoded_smiles = []
        for i in smiles_array:
            enc_smile = np.argmax(i, axis=1)
            encoded_smiles.append(enc_smile)
        return encoded_smiles

    def load_dataset(self, path, samples):
        df = pd.read_csv(path, nrows=samples)
        return df['smiles'].tolist()
