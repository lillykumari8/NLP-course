# CSE 517 - HW 4 #

* Changes made to the model.py and run.py (to add model 'att_lstm') scripts

* For CBOW, cmd: python cli.py --train --model cbow --out_dir out/cbow --device_type gpu

* For RNN (GRU, keep_prob 0.9), cmd: python cli.py --train --model rnn --out_dir out/rnn_gru_keep_prob_09 --device_type gpu --keep_prob 0.9

* For RNN (GRU, keep_prob 0.7), cmd: python cli.py --train --model rnn --out_dir out/rnn_gru_keep_prob_07 --device_type gpu --keep_prob 0.7

* For RNN (GRU, keep_prob 0.5), cmd: python cli.py --train --model rnn --out_dir out/rnn_gru_keep_prob_05 --device_type gpu --keep_prob 0.5 

* For Attention_RNN (GRU, keep_prob 0.9), cmd: python cli.py --train --model att --out_dir out/att_gru_keep_prob_09 --device_type gpu --keep_prob 0.9

* For Attention_RNN (LSTM, keep_prob 0.9, max_context_size 200), cmd: python cli.py --train --model att_lstm --out_dir out/att_lstm_keep_prob_09_con200 --device_type gpu --keep_prob 0.9  --max_context_size 200