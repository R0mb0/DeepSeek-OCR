# Configurazione progetto DeepSeek-OCR (versione con valori più conservativi per evitare OOM)
# Modifica i valori qui sotto se vuoi usare impostazioni più aggressive (maggiore concorrenza / più crop).
# Nota: imposto valori più bassi per MAX_CONCURRENCY, NUM_WORKERS e MAX_CROPS per ridurre l'uso di VRAM/CPU.

# Dimensioni visive (lascia se non sei sicuro — modificarle può influire su qualità e compatibilità)
BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True

# Numero minimo/massimo di crop (riduci MAX_CROPS se hai poca VRAM)
MIN_CROPS = 1
MAX_CROPS = 4  # ridotto da 6 a 4 per ridurre il numero di token visivi e l'uso di memoria

# Concorrenza e workers (valori conservativi per macchine con poca VRAM/CPU)
MAX_CONCURRENCY = 4   # numero massimo di richieste concorrenti (ridotto da 100)
NUM_WORKERS = 4       # workers per preprocessing immagini (ridotto da 64)

# Stampe e opzioni
PRINT_NUM_VIS_TOKENS = False
SKIP_REPEAT = True

# Percorsi e modello
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'  # Cambia con il path locale se usi snapshot locale
INPUT_PATH = ''
OUTPUT_PATH = ''

# Prompt predefinito usato dallo script
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'

# NOTE IMPORTANTI SULL'AMBIENTE
# - Assicurati di impostare queste variabili d'ambiente PRIMA di eseguire qualsiasi script Python:
#     export TOKENIZERS_PARALLELISM=false
#     export CUDA_VISIBLE_DEVICES=0
#     export VLLM_USE_V1=0
# - Se riscontri frammentazione / OOM, prova anche:
#     export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
#
# - Se la GPU è molto piccola (ad es. <6GB), imposta MAX_CONCURRENCY=1 e MAX_CROPS=1 per minimizzare l'uso VRAM.

from transformers import AutoTokenizer

# Creazione eager del tokenizer.
# Nota: alcuni componenti del progetto si aspettano un tokenizer concreto al momento
# dell'import. Se preferisci evitare il download automatico qui, puoi trasformare
# questa inizializzazione in lazy (usando get_tokenizer() che fa il load on-demand).
#
# AVVERTENZA: questa chiamata può effettuare I/O (download) se lo snapshot non è in cache.
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

def get_tokenizer():
    """Return the tokenizer instance (compatibilità)."""
    return TOKENIZER
