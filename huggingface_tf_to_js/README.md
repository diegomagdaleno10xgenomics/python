# Tensorflow HuggingFace to Javascript

Description: This is a small project based on the Tensorflow blog post from Huggingface detailing how to use their transformers library to host lightweight models in node.js applications.

Source: [Tensorflow Blog post](https://blog.tensorflow.org/2020/05/how-hugging-face-achieved-2x-performance-boost-question-answering.html)
Source: [TensorflowJS node setup](https://www.tensorflow.org/js/tutorials/setup)
Source: [Huggingface node documentation](https://github.com/huggingface/tokenizers/tree/main/bindings/node)

Check the resulting SavedModel contains the correct signature:
`saved_model_cli show --dir tf_distilbert_cased_savedmodel --tag_set serve --signature_def serving_default`

Expected output from command:
```
The given SavedModel SignatureDef contains the following input(s):
  inputs['attention_mask'] tensor_info:
   dtype: DT_INT32
   shape: (-1, 384)
   name: serving_default_attention_mask:0
  inputs['input_ids'] tensor_info:
   dtype: DT_INT32
   shape: (-1, 384)
   name: serving_default_input_ids:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['output_0'] tensor_info:
   dtype: DT_FLOAT
   shape: (-1, 384)
   name: StatefulPartitionedCall:0
  outputs['output_1'] tensor_info:
   dtype: DT_FLOAT
   shape: (-1, 384)
   name: StatefulPartitionedCall:1
Method name is: tensorflow/serving/predict
```
