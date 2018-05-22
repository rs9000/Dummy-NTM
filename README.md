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
<img src="pics/loss.jpg" width="600">
