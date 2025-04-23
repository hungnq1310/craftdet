from refinet.tool.predictor import Predictor
from refinet.tool.config import Cfg
import torch
from pathlib import Path
import os
config = Cfg.load_config_from_name('vgg_seq2seq')
config['weights'] = Path("/kaggle/input/weight-craft/weights/ocr/vgg_seq2seq.pth").expanduser().resolve()
config['device'] = 'cuda:0'
# ocr = Predictor(config)
# save_ocr_path = Path(os.getcwd()+ "/prediction/ocr").expanduser().resolve()

from refinet.model.transformerocr import VietOCR
from refinet.model.vocab import Vocab

vocab = Vocab(config['vocab'])
device = config['device']

model = VietOCR(len(vocab),
    config['backbone'],
    config['cnn'],
    config['transformer'],
    config['seq_modeling']
)
model.load_state_dict(torch.load(config['weights'] , map_location=torch.device(config['device'])))


# ----------------------------------------CONVERT-------------------------------------------

# CNN part
def convert_cnn_part(img, save_path, model, max_seq_length=128, sos_token=1, eos_token=2):
    with torch.no_grad():
        src = model.cnn(img)
        torch.onnx.export(model.cnn,
                          img,
                          save_path,
                          export_params=True,
                          do_constant_folding=True,
                          verbose=True,
                          input_names=['IMAGE'],
                          output_names=['OUTPUT'],
                          dynamic_axes={'IMAGE': {0: 'batch'},
                                        'OUTPUT': {1: 'batch'}})  # channel output of cnn in the 2nd dimension

    return src


# Encoder part
def convert_encoder_part(model, src, save_path):
    encoder_outputs, hidden = model.transformer.encoder(src)
    torch.onnx.export(model.transformer.encoder,
                      src,
                      save_path,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['src'],
                      output_names=['encoder_outputs', 'hidden'],
                      dynamic_axes={'src': {0: "channel_input"},
                                    'encoder_outputs': {0: 'channel_output'}})
    return hidden, encoder_outputs


# Decoder part
def convert_decoder_part(model, tgt, hidden, encoder_outputs, save_path):
    tgt = tgt[-1]
    print(tgt)
    torch.onnx.export(model.transformer.decoder,
                      (tgt, hidden, encoder_outputs),
                      save_path,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['tgt', 'hidden', 'encoder_outputs'],
                      output_names=['output', 'hidden_out', 'last'],
                      dynamic_axes={'encoder_outputs': {0: "channel_input"},
                                    'last': {0: 'channel_output'}})


# Export model
img = torch.rand(1, 3, 512, 512).to(torch.float32)  # input tensor of torch model (N x C x H x W)
src = convert_cnn_part(img, '/kaggle/working/model_cnn1.onnx', model)
print(0)
hidden, encoder_outputs = convert_encoder_part(model, src, '/kaggle/working/model_encoder.onnx')
print(1)
device = img.device
tgt = torch.LongTensor([[1] * len(img)]).to(device)
convert_decoder_part(model, tgt, hidden, encoder_outputs, '/kaggle/working/model_decoder.onnx')
print(2)

# ----------------------------------------------------------------------------------------------------