# Checklist on AllenNLP semantic role labelling model (SRL)
Individual Assignment for the Course NLP Technology III at VU University Amsterdam

## AUTHORS
------------------
S.Shen (Shuyi Shen)

## PROJECT STRUCTURE
-------------------
While testing individual components is a common practice in software engineering, modern NLP models are rarely built one component at a time. With this in mind, this project aims to identify critical failures for semantic role labeling (SRL) models by testing the core capabilities from linguistic aspect using Checklist tools. 

**What you will find in this project:**
- Lexicalization Test(appropriately understanding named entities)
- PP Attachment Test(to ARG0 and ARG1 in prepositional phrase)
- Argument alteration test, Robustness (to typos, irrelevant changes, etc)
- Long-range Dependency Test(to ARG1 with parathesis)
- Syntactic Variation Test (to ARG0 and ARG1 with passive/active and questions/statements)
- Label Confusion Test ( to ARG2 with AM-DIR, AM-LOC and AM-MNR)

## Example of the experiment
----------------------------
- Test1: Lexicalization Test
  - Capability: Lexicalization of Arguments 
  - Test Type: Minimum Functionality (MFT)

Neural models such as RnnOIE (Stanovsky et al., 2018), built on a static vocabulary and word embedding, often fails to identify some rare words as semantic roles. With this in mind, we are motivated to design the test by replacing part of ordinary agents, instruments, and patients with those appearing less frequently in the dataset. This experiment aims to check if the model can appropriately identify proper names (common/uncommon, western, non-western) since most NLP applications are established on a fixed vocabulary, which means OOV (out of vocabulary) often occurs when some rare or low-frequency words, function as specific semantic roles. (e.g., “Kim” as the agent / ARG-0 in the example “Kim was reading”) We expect SRL systems could be able to perform well in these cases.

## Models
----------
In this section, we chose two neural systems of different architectures to evaluate on our challenge dataset: the LSTM-based model RnnOIE (Stanovsky et al., 2018) and the BERT-based model (Shi & Lin, 2019), both of which are introduced by AllenNLP. Concerning the inner structure of LSTM, the model is expected to capture long-distance dependency. In terms of BERT, the model is expected to capture syntactic information by the contextual representation of the sentence <[CLS] sentence [SEP]> concatenated with predicate indicator embeddings.

## Experimental Result: testing pre-trained SRL models with CheckList
---------------------------------------------------------------------

|      |Test TYPE | Failure rate - LSTM / BERT| Example Test cases (with expected behavior and prediction) |
| --- | --- | --- | --- |
| Label Confusion Test | MFT | 40% / 100% | **Sentence:** The worker move the containers into that building with a crane., <br /> **target:** with a crane, <br />**expected*:** ARG2, <br />**prediction:** ARGM-MNR |
| Long Range Test | MFT: ARG0 and ARG1 with Parenthesis | 0% / 0% | **Sentence:** The woman, though in a depression, actually offered a good explanation, <br /> **target:** ARG0 and ARG1, <br />**expected*:** ["The woman", "a good explanation"], <br /> **prediction**: ["The woman", "a good explanation"] |
| LexicalTest | MFT:ARG1 with Vietnamese names | 10% / 20% | **Sentence:** Someone killed Emmanuel Dương last night., <br /> **target:**  ARG1, <br />**expected*:** "Emmanuel Dương", <br /> **prediction:** ["Dương", "Emmanuel"] |
| PP Attachment Test | MFT: /ARG2/Instrument/ARGM /ARG1/in prepositional phrase | 0% / 0% | **Sentence:** The spy saw the cop with the revolver <br /> **target:** with the revolver, <br />**expected*:** ARG1, <br />**prediction:** ARG1|
| Argument Alter Test | INV | 100% / 100% | **Sentence:** “Jack sprayed paint on the wall.”, “Jack sprayed the wall with paint.” <br /> **target:** INV <br />**expected*:** ["Jack", "paint"], <br />**prediction:** ["Jack", "the wall"], |
| Robustness Test | INV: Test ARG1 on typos | 60% / 40% | **Sentence:** "The boy broke the vase.", "The boy broke th evase." <br /> **target:** INV <br />**expected*:** ["B-ARG0", "I-ARG0", "B-V", "B-ARG1", "I-ARG1", "O"] <br />**prediction:** ["B-ARG1", "I-ARG1", "B-V", "O", "O", "O"] |
| Syntax Variation Test | INV:Test ARG0 and ARG1 on passive/active & statement/ questions sentences | 80% / 80% | **Sentence:** "Pia gives the lecture.", "the lecture is given by Pia." <br /> **target:** INV <br />**expected*:** ["Pia", "the lecture"], <br />**prediction:** ["by Pia", "the lecture"],|

### HOW TO USE IT
-------------------
- STEP 1. Install behavioral testing libraries **checklist**, **AllenNLP** module, and
  spacy **en_core_web_sm** by running:
  - `pip install checklist`, 
  - `pip install AllenNLP`, 
  - `python -m spacy download en_core_web_sm`
  
   <br />  
- STEP 2. Go to **Checklist_SRL/checklist_srl** folder, choose one of the
    linguistic phenomenon <br /> that you want to experiment and run **main.py**

