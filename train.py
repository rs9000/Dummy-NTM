import torch
import numpy as np
import argparse
from ntm import NTM
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from function_dataset import FunctionDataset
import matplotlib
import matplotlib.cm

def clip_grads(net, args):
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(args.min_grad, args.max_grad)


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_size', type=int, default=10, help='The length of input vector', metavar='')
    parser.add_argument('--output_size', type=int, default=10,
                        help='The The length of output vector', metavar='')
    parser.add_argument('--memory_capacity', type=int, default=50,
                        help='Number of vectors stored in memory', metavar='')
    parser.add_argument('--function_size', type=int, default=9,
                        help='Dimensionality of functions to learn', metavar='')
    parser.add_argument('--n_functions', type=int, default=5,
                        help='Number of functions', metavar='')
    parser.add_argument('--controller_type', type=str, default="feed",
                        help='Type of controller: feedforward, rnn, rnn_seq2seq', metavar='')
    parser.add_argument('--controller_dim', type=int, default=256,
                        help='Dimensionality of the feature vector produced by the controller', metavar='')
    parser.add_argument('--max_program_length', type=int, default=5,
                        help='Max number of functions', metavar='')
    parser.add_argument('--training_samples', type=int, default=999999,
                        help='Number of training samples', metavar='')
    parser.add_argument('--use_curriculum', type=float, default=False,
                        help='Using curriculum in training', metavar='')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Optimizer learning rate', metavar='')
    parser.add_argument('--min_grad', type=float, default=-10,
                        help='Minimum value of gradient clipping', metavar='')
    parser.add_argument('--max_grad', type=float, default=10,
                        help='Maximum value of gradient clipping', metavar='')
    parser.add_argument('--logdir', type=str, default='./logs2',
                        help='The directory where to store logs', metavar='')
    parser.add_argument('--loadmodel', type=str, default='',
                        help='The pre-trained model checkpoint', metavar='')
    parser.add_argument('--savemodel', type=str, default='checkpoint',
                        help='Name/Path of model checkpoint', metavar='')
    parser.add_argument('--eval', type=bool, default=False,
                        help='Evaluate primitive functions', metavar='')

    return parser.parse_args()

def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for Torch/Numpy that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: Matplotlib default colormap)

    Returns a 4D uint8 tensor of shape [height, width, 4].
    """

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    value = value.squeeze()

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    return value


def eval(model, dataset, args):
    sample, _, _ = dataset[2]
    sample = torch.tensor(sample)
    program = torch.zeros(1, args.n_functions, args.n_functions)
    for i in range(args.n_functions):
        program[0, i, i] = 1
    model(sample.cuda(), program.cuda())
    memory, _, ntm_programs = model.get_memory_info()
    gen_programs = dataset.program_list()

    test = np.zeros((50, 5))
    for j, memory in enumerate(memory):
         for i, gen_program in enumerate(gen_programs):
                gen_program = gen_program.view(1, args.function_size*args.function_size).cuda()
                memory = memory.view(1, -1)
                pdist = torch.nn.CosineSimilarity()
                dist = pdist(gen_program, memory)
                test[j][i] = dist.data
                #print(ntm_program)
                #print("Sim [" + str(i) + "][" + str(j) + "] = " + str(dist.data))

    print(test.round(2))
    return


def train():
    args = parse_arguments()
    writer = SummaryWriter()
    dataset = FunctionDataset(input_vector_size=args.input_size,
                              function_vector_size=args.function_size,
                              output_vector_size=args.output_size,
                              n_functions=args.n_functions,
                              max_program_length=args.max_program_length,
                              samples_per_epoch=args.training_samples,
                              use_curriculum=args.use_curriculum,
                              train_transforms=None)

    model = NTM(M=args.memory_capacity,
                N=args.function_size*args.function_size,
                num_inputs=args.input_size,
                num_outputs=args.output_size,
                function_vector_size=args.function_size,
                n_functions=args.n_functions,
                use_RnnController=args.controller_type,
                controller_dim=args.controller_dim,
                input_embedding=dataset.input_embedding,
                output_embedding=dataset.output_embedding,
                ).cuda()

    print(model)
    entropy = dataset.entropy(10000, 10, False)
    print("Entropy of ground truth vectors: " + str(entropy))

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("--------- Number of parameters -----------")
    print(model.calculate_num_params())
    print("--------- Start training -----------")

    losses = []

    if args.loadmodel != '':
        model.load_state_dict(torch.load(args.loadmodel))

    if args.eval:
        eval(model, dataset, args)
        return

    for e in range(len(dataset)):
        X, program, Y = dataset[e]
        optimizer.zero_grad()

        program = torch.unsqueeze(program,0)
        program.requires_grad = True

        y_pred = model(X.cuda(), program.cuda())

        loss = criterion(y_pred, Y.cuda())
        loss.backward()
        clip_grads(model, args)
        optimizer.step()
        losses += [loss.item()]

        if e % 200 == 0:
            mean_loss = np.array(losses[-200:]).mean()
            print("Loss: ", loss.item())
            writer.add_scalar('Mean loss', mean_loss, e)
            losses = []
            if e % 200 == 0:
                print(y_pred)
                print(Y)
                mem_pic, read_pic, ntm_programs = model.get_memory_info()
                gen_programs = dataset.program_list()
                pic1 = vutils.make_grid(y_pred, normalize=True, scale_each=True)
                pic2 = vutils.make_grid(Y, normalize=True, scale_each=True)
                pic3 = vutils.make_grid(mem_pic, normalize=True, scale_each=True)
                pic4 = vutils.make_grid(read_pic, normalize=True, scale_each=True)
                writer.add_image('NTM output', colorize(pic1.data), e)
                writer.add_image('True output', colorize(pic2), e)
                writer.add_image('Memory', pic3, e)
                writer.add_image('Read addressing', pic4, e)

                for i, ntm_program in enumerate(ntm_programs):
                    pic6 = vutils.make_grid(ntm_program.view(args.function_size, args.function_size), normalize=True, scale_each=True)
                    writer.add_image('NTM software learned ' + str(i), pic6, e)

                for i, gen_program in enumerate(gen_programs):
                    pic7 = vutils.make_grid(gen_program, normalize=True,
                                            scale_each=True)
                    writer.add_image('True software ' + str(i), pic7, e)

                torch.save(model.state_dict(), args.savemodel + "_" + args.controller_type + ".model")


if __name__ == "__main__":
    train()
