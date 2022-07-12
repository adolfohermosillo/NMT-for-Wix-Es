import re

def normwix(text):
    text = text.lower()
    text = re.sub(r"[`´‘’ʔ']", "'", text, flags=re.IGNORECASE)
    text = re.sub(r"'", "ʔ", text, flags=re.IGNORECASE)
    text = re.sub(r" +", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[üïɨ+]", "ɨ", text, flags=re.IGNORECASE)
    text = re.sub(r"ḱ", "k", text, flags=re.IGNORECASE)
    text = re.sub(r"(ẃ|ẁ)", "w", text, flags=re.IGNORECASE)
    text = re.sub(r"[ń]", "n", text, flags=re.IGNORECASE)
    text = re.sub(r"[áàäá]", "a", text, flags=re.IGNORECASE)
    text = re.sub(r"[éèëéë́]", "e", text, flags=re.IGNORECASE)
    text = re.sub(r"[íìií]", "i", text, flags=re.IGNORECASE)
    text = re.sub(r"[óòöó]", "o", text, flags=re.IGNORECASE)
    text = re.sub(r"[úùú]", "u", text, flags=re.IGNORECASE) 
    return text


def aggressive_normwix(text):
    text.lower()
    text = normwix(text)
    text = re.sub(r"([a-z+])\1+", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r" ʔ", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"v", "w", text, flags=re.IGNORECASE)
    text = re.sub(r"(c|qu)", "k", text, flags=re.IGNORECASE)
    #text = re.sub(r"[0-9]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"ch", "ts", text, flags=re.IGNORECASE)
    text = re.sub(r"rr", "x", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!t|\[)s", "ts", text, flags=re.IGNORECASE)
    text = re.sub(r"([a-z+])\1+", r"\1", text, flags=re.IGNORECASE)
    return text
  
def tokenize(text):
    text = re.sub(r"(?<![\s])([\)|\(|.|,|,\-,\"|:|;|¿|?|¡|!|«|»|—|―])", r" \1", text)
    text = re.sub(r"([\)|\(|.|,|,\-,\"|:|;|¿|?|¡|!|«|»|—|―])(?<![\s])", r"\1 ", text)
    text = re.sub(r"(ç|_)",'',text, flags=re.IGNORECASE)
    text = re.sub(r"    ",' ',text, flags=re.IGNORECASE)
    text = re.sub(r"^ ", "", text, flags=re.IGNORECASE)
    return text
  
def normes(text):
    return text
    
  
