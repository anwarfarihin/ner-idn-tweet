1. Fitur-fitur testing data sudah tersedia di sini dalam folder /ner/test/ner
2. Model untuk 5 skenario ada di dalam folder /ner/model/
3. Contoh hasil output dalam bentuk .json ada di folder /ner/output. File .txt dalam bentuk enamex di dapatkan dengan script post process di luar direktori ini, yaitu di ../postprocess

Untuk mencoba predict dan menghasilkan file json, pertama set model dan skenario yang diinginkan pada file /ner/config.ini seperti sebagai berikut

[general]
scenario: ner

[ner]
project: ner
model_name: ner_1
we_model: we_ner_2
we_dir: word-embedding/model
training_data: ./data/${project}/data.json
testing_data: ./data/${project}/test.json
x_train_out: ./features/${project}/x_train.json
y_train_out: ./features/${project}/y_train.json
x_test_out: ./features/${project}/x_test.json
y_test_out: ./features/${project}/y_test.json
context: False
pos: True
wordshape: False


-Ubah model_name dari ner_1 menjadi apapun yang ada di folder /ner/model/
-context, pos, dan wordshape apakah dinyalakan atau dimatikan (False atau True), juga diatur di file config ini.
-Khusus untuk ner_1, setting yang digunakan adalah hanya POS yang True.
-Model lain setting fiturnya sesuai dengan nama file nya. Misalnya ner_we artinya hanya menggunakan fitur Word Embedding, maka context, pos, wordshape False semua.

-jalankan python predict.py dan file .json sesuai dengan model_name akan muncul pada /ner/output