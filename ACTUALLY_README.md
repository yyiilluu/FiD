## format dataset
list of dict in json.  

Entry example:
```
{
  'id': '0',
  'question': 'What element did Marie Curie name after her native land?',
  'target': 'Polonium',
  'answers': ['Polonium', 'Po (chemical element)', 'Po'],
  'ctxs': [
            {
                "title": "Marie Curie",
                "text": "them on visits to Poland. She named the first chemical element that she discovered in 1898 \"polonium\", after her native country. Marie Curie died in 1934, aged 66, at a sanatorium in Sancellemoz (Haute-Savoie), France, of aplastic anemia from exposure to radiation in the course of her scientific research and in the course of her radiological work at field hospitals during World War I. Maria Sk\u0142odowska was born in Warsaw, in Congress Poland in the Russian Empire, on 7 November 1867, the fifth and youngest child of well-known teachers Bronis\u0142awa, \"n\u00e9e\" Boguska, and W\u0142adys\u0142aw Sk\u0142odowski. The elder siblings of Maria"
            },
            {
                "title": "Marie Curie",
                "text": "was present in such minute quantities that they would eventually have to process tons of the ore. In July 1898, Curie and her husband published a joint paper announcing the existence of an element which they named \"polonium\", in honour of her native Poland, which would for another twenty years remain partitioned among three empires (Russian, Austrian, and Prussian). On 26 December 1898, the Curies announced the existence of a second element, which they named \"radium\", from the Latin word for \"ray\". In the course of their research, they also coined the word \"radioactivity\". To prove their discoveries beyond any"
            }
          ]
}
```
## Env
```shell
/home/yilu/FiD
source venv/bin/activate
```
and run the command 

## training
```shell
python train_reader.py \
    --train_data open_domain_data/NQ/train.json \
    --eval_data open_domain_data/NQ/dev.json \
    --model_size small \
    --per_gpu_batch_size 1 \
    --n_context 10 \
    --name test_exp \
    --checkpoint_dir checkpoint
```

## inference
```shell
python inference.py \
        --model_path checkpoint/test_exp/checkpoint/best_dev \
        --eval_data open_domain_data/NQ/test.json \
        --per_gpu_batch_size 1 \
        --n_context 10 \
        --name eval_test \
        --checkpoint_dir checkpoint
```

or we could provide question and passages by calling the function