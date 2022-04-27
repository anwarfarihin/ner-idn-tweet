# ner-idn-tweet

## Data

Dataset folder
- iter: Data raw yang dipisahkan untuk dianotasi per iterasi (terdapat 3 iterasi).
- kappa_data: Data yang sudah dianotasi namun belum digabungkan.
- raw: Data yang belum dilabeli. `NER_final.csv` berisi seluruh dataset mentah yang akan digunakan dalam project ini.
- simpletransformers: data dalam format simpletransformers, menggunakan 6 labels. Dipisahkan menggunakan stratified KFold.
- simpletransformers_4: data dalam format simpletransformers, menggunakan 4 labels. Dipisahkan menggunakan stratified KFold.
- StratifiedKFold: data sudah dipisahkan pertweet, menggunakan stratified KFold

>Tag 6 Label: Person, Location, Organization, Product, Work of Art, Event  

Pada akhirnya, karena `Work of Art` dan `Event` memiliki jumlah tag yang minim, dilakukan pengurangan jumlah tag dari 6 menjadi 4 (Work of Art dan Event dibuang)  

>Tag 4 Label: Person, Location, Organization, Product
## Model
Model yang digunakan:
- indobert-base-uncased
- indobert-base-p1
- indobert-large-p1
- indobert-large-p2
- bert-base-cased 
- bert-base-multilingual-cased

Rekomendasi pretrained model: [indobertweet](https://github.com/indolem/IndoBERTweet)
Model retrain code dir: Model/simpletransformers/

# TO DO
1. Retrain model menggunakan data dengan 4 label, komparasi model (performa, waktu, memory usage) 
1. Deployment (Python + Pubsub + Docker)
3. Masih ada beberapa tag ambigu (contoh gabungan gelar & nama:”presiden joko widodo meresmikan…” → presiden ada yg tag PER ada yg enggak), perlu dianalisis konsistensinya
4. BERT inggris lebih tinggi dibanding indoBERT --> micro analysis

