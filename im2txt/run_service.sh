# Run inference to generate captions.
bazel run -c opt im2txt/run_inference_service -- \
  --checkpoint_path="/root/im2txt_pretrained/model.ckpt-2000000" \
  --vocab_file="/root/im2txt_pretrained/word_counts.txt" \
