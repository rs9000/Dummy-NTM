# Simple-DNTM
Not a Neural turing machine

### How to use

```
usage: train.py [-args]

```

### Model:<br>
<img src="pics/dntm.jpg" width="700">

## Controller type
--controller_type = [feedforward, rnn, rnn_seq2seq] <br><br>
<img src="pics/controller_types.jpg" width="500">


## Learn functions task <br>
- Input Vector Length: 10<br> 
- Max Program length: 5<br> 
- N° Functions: 5<br> 
- Function size: 9x9<br> 
<img src="pics/out.jpg" width="600">

### Memory snapshot:
<img src="pics/memory.jpg" width="300">

### Addressing locations:
<img src="pics/addr.jpg" width="300">

### Sequence of learned functions applied by the model to generate this output: <br>
<img src="pics/softw.jpg" width="600">

### True primitive functions (never seen by model): <br>
<img src="pics/primitive.jpg" width="600">

### Loss: <br>
FeedForward Controller <br>
<img src="pics/loss.jpg" width="600">

Rnn Controller <br>
<img src="pics/dntmv2-loss-rnn.jpg" width="600">

### Similarity between learned functions and true functions
FeedForward Controller <br>
- Similarity Function[0] = tensor([ 0.9816], device='cuda:0')
- Similarity Function[1] = tensor([ 0.9695], device='cuda:0')
- Similarity Function[2] = tensor([ 0.9227], device='cuda:0')
- Similarity Function[3] = tensor([ 0.9300], device='cuda:0')
- Similarity Function[4] = tensor([ 0.9739], device='cuda:0')

RNN Controller <br>
- Similarity Function[0] = tensor([ 0.9820], device='cuda:0')
- Similarity Function[1] = tensor([ 0.9802], device='cuda:0')
- Similarity Function[2] = tensor([ 0.9776], device='cuda:0')
- Similarity Function[3] = tensor([ 0.9707], device='cuda:0')
- Similarity Function[4] = tensor([ 0.9797], device='cuda:0')

RNN_seq2seq Controller <br>
- Similarity Function[0] = tensor([-0.0936], device='cuda:0')
- Similarity Function[1] = tensor([ 0.1777], device='cuda:0')
- Similarity Function[2] = tensor([-0.1399], device='cuda:0')
- Similarity Function[3] = tensor([ 0.1160], device='cuda:0')
- Similarity Function[4] = tensor([ 0.4007], device='cuda:0')
