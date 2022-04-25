// import { BertWordPieceTokenizer } from "tokenizers";
const BertWordPieceTokenizer = require("tokenizers").BertWordPieceTokenizer;
const tf = require("@tensorflow/tfjs");

// Optional load the binding:
//require("@tensorflow/tfjs-node-gpu"); // if running GPU.
require("@tensorflow/tfjs-node");

// Load the model located in the path.
const path = '../tf_distilbert_cased_savedmodel';
//const model = await tf.node.loadSavedModel(path);
// const model = tf.node.loadSavedModel(path);
const model = tf.loadGraphModel(path);

const result = tf.tidy(() => {
	// IDs and AttentionMask are of type number[][].
	const inputTensor = tf.tensor(ids, undefined, "int32");
	const maskTensor = tf.tensor(attentionMask, undefined, "int32");

	// Run model inference.
	return model.predict({
		// "input_ids" and "attention_mask" correspond to the names specified
		// in the signature passed to get_concrete_function during the model
		// conversion.
		"input_ids": inputTensor,
		"attention_mask": maskTensor,
	}); // as tf.NamedTensorMap;
});

// Extract the start and end logits from the tensors returned by
// model.predict().
const [startLogits, endLogits] = Promise.all([ // await Promise.all([
	result["output_0"].squeeze().array(), // as Promise,
	result["output_1"].squeeze().array() // as Promise,
]);

// Clean up memory used by the result tensor signature since we dont
// need it anymore.
tf.dispose(result);

// Initializer.
const vocabPath = "../huggingface-distilbert-pretrained";
const tokenizer = BertWordPieceTokenizer.fromOptions({
	vocabFile: vocabPath, lowercase:false
}); 

// 384 matches the shape of the input signature provided while
// exporting to SavedModel.
tokenizer.setPadding({maxLength:384});

// Here question and context are in their original string format.
// const encoding = await tokenizer.encode(question, context);
const encoding = tokenizer.encode(question, context);
const { ids, attentionMask } = encoding;
