# Entity Projection via Machine Translation for Cross-Lingual NER
Code for (this)[https://arxiv.org/pdf/1909.05356.pdf] paper. We demonstrate 
that using off-the-shelf Machine Translation (MT) systems and a few simple 
heuristics, significant gains can be made towards cross-lingual NER for 
<em>medium-resource languages</em>[^1].

[^1]: We define medium-resource languages to be those for which while strong 
off-the-shelf MT systems exist, large annotated corpora for NER do not.

# Set-up

## Environment
This code has been written in Python 3.6. Please create a dedicated 
environment (using virtualenv or conda) and install the packages listed in 
the requirements file in your environment using the following command. Note 
that all the commands listed in this README assume that you are in the parent
 project directory, i.e., in `cross_lingual_ner/`.

```
pip install -r requirements.txt
```

## Using Google Cloud Translation API
The Google Cloud Translation API needs to be used twice in order to 
successfully run the TMP method:

<ol>
<li>To translate sentences from source (language) to target.</li>
<li>To translate each entity phrase in a source sentence to target.</li>
</ol>

Please find below instructions to set up and use the API.

### Setting up the API
Please follow the steps listed [here](https://cloud.google.com/translate/docs/advanced/setup-advanced)
to set up the API. Once your setup is finished, you will have access to an 
API Key that would be required to authenticate during API usage. Store this 
key (string) in a text file (not JSON) in your project directory. Please ensure 
that this key remains private to avoid unauthorized usage from your account.

### Using the API programmatically
The following function in `src/util/tmp.py` accesses the Translation API:

```
from googleapiclient.discovery import build
.
.
.
def get_google_translations(src_list, src_lang_code, tgt_lang_code, api_key):
    service = build('translate', 'v2', developerKey=api_key)
    tgt_dict = service.translations().list(
        source=src_lang_code, target=tgt_lang_code, q=src_list).execute()
    return [t['translatedText'] for t in tgt_dict['translations']]
```

#### Note
The Google Cloud Translation service can at times error out due to
 request arrival rate exceeding the maximum rate allowed. The argument 
`batch_size` (in `main.py`) has been set to 128 and `time_sleep` to 10 (seconds)
to minimize such errors. Note that these arguments are used only while 
translating sentences. Entity phrase translation occurs on a 
sentence-by-sentence basis without batching and without any wait time.

However, despite these measures, these errors continue to occur. If that 
happens, please note down the index of the batch (while translating sentences) 
and that of the sentence (while translating entity phrases) at which the error 
occurs. If the error occurs while translating sentences, re-run the process 
with the argument `sent_iter` (in `main.py`) set to this index. If the error 
occurs while translating entity do this with the argument `phrase_iter`.

Initially, both these indices are set to -1, so that all sentences or 
entity phrases get sent to the API for translation. When one or both of these 
have positive integral values, the batches or sentences numbered lower than 
these indices (`sent_iter`, `phrase_iter`) are not sent again for translation.