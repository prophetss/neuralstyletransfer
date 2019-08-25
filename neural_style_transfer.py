import d2lzh as d2l
from mxnet import autograd, gluon, image, init, nd
from mxnet.gluon import model_zoo, nn
import time
import os
import argparse

parser = argparse.ArgumentParser(description='图片样式迁移')
parser.add_argument('content_path', type=str, help='内容图片路径')
parser.add_argument('style_path', type=str, help='样式图片路径')
parser.add_argument('out_path', type=str, help='输出图片路径')
parser.add_argument('--out_img_shape', type=str, default="600*400", help='输出图片尺寸')
parser.add_argument('--style_save_path', type=str, default=None, help='样式结果保存路径，保存后下次可直接读取使用')
parser.add_argument('--max_epochs', type=int, default=300, help='训练迭代次数')
parser.add_argument('--content_weight', type=float, default=0.1, help='内容损失权重')
parser.add_argument('--style_weight', type=float, default=1e5, help='样式损失权重')
parser.add_argument('--tv_weight', type=float, default=10, help='噪音损失权重')

args = parser.parse_args()
print(args)

content_path = args.content_path
style_path = args.style_path
out_path = args.out_path
out_img_shape = args.out_img_shape
style_save_path = args.style_save_path
max_epochs = args.max_epochs
content_weight = args.content_weight
content_path = args.content_path
style_weight = args.style_weight
tv_weight = args.tv_weight

# 输入图像标准化,数据来自Imagenet数据集.
rgb_mean = nd.array([0.485, 0.456, 0.406])
rgb_std = nd.array([0.229, 0.224, 0.225])


def pre_process(img, image_shape):
    img = image.imresize(img, *image_shape)
    img = (img.astype('float32') / 255 - rgb_mean) / rgb_std
    return img.transpose((2, 0, 1)).expand_dims(axis=0)


def post_process(img):
    img = img[0].as_in_context(rgb_std.context)
    return (img.transpose((1, 2, 0)) * rgb_std + rgb_mean).clip(0, 1)


# 选取vgg19的block1到5的第一个卷积层提取样式特征，block4的最后一个卷积层提取内容特征
style_layers, content_layers = [0, 5, 10, 19, 28], [25]

pretrained_net = model_zoo.vision.vgg19(pretrained=True)
net = nn.Sequential()
for i in range(max(content_layers + style_layers) + 1):
    net.add(pretrained_net.features[i])
ctx = d2l.try_gpu()
net.collect_params().reset_ctx(ctx)


def extract_features(X, content_layers, style_layers):
    # 保存样式和内容所需层的特征结果
    contents, styles = [], []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles


def get_contents(content_img, image_shape, ctx):
    content_X = pre_process(content_img, image_shape).copyto(ctx)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y


def get_styles(style_img, image_shape, ctx):
    style_X = pre_process(style_img, image_shape).copyto(ctx)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y


def content_loss(Y_hat, Y):
    return (Y_hat - Y).square().mean()


def gram(X):
    num_channels, n = X.shape[1], X.size // X.shape[1]
    X = X.reshape((num_channels, n))
    return nd.dot(X, X.T) / (num_channels * n)


def style_loss(Y_hat, gram_Y):
    return (gram(Y_hat) - gram_Y).square().mean()


def tv_loss(Y_hat):
    return 0.5 * ((Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).abs().mean() +
                  (Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).abs().mean())


def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # 分别计算内容损失、样式损失和总变差损失
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # 对所有损失求和
    l = nd.add_n(*styles_l) + nd.add_n(*contents_l) + tv_l
    return contents_l, styles_l, tv_l, l


class GeneratedImage(nn.Block):
    def __init__(self, img_shape, **kwargs):
        super(GeneratedImage, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=img_shape)

    def forward(self):
        return self.weight.data()


def get_inits(X, ctx, lr, styles_Y):
    gen_img = GeneratedImage(X.shape)
    gen_img.initialize(init.Constant(X), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(gen_img.collect_params(), 'adam',
                            {'learning_rate': lr})
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer


def train(X, contents_Y, styles_Y, ctx, lr, max_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, ctx, lr, styles_Y)
    for i in range(max_epochs):
        start = time.time()
        with autograd.record():
            contents_Y_hat, styles_Y_hat = extract_features(
                X, content_layers, style_layers)
            contents_l, styles_l, tv_l, l = compute_loss(
                X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step(1)
        nd.waitall()
        if i % 50 == 0 and i != 0:
            print('epoch %3d, content loss %.2f, style loss %.2f, '
                  'TV loss %.2f, %.2f sec'
                  % (i, nd.add_n(*contents_l).asscalar(),
                     nd.add_n(*styles_l).asscalar(), tv_l.asscalar(),
                     time.time() - start))
        if i % lr_decay_epoch == 0 and i != 0:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
            print('change lr to %.1e' % trainer.learning_rate)
    return X


lr, lr_decay_epoch = 0.01, 100


def process(content_path, style_path, output_shape, style_save_path):
    content_img, style_img = image.imread(content_path), image.imread(style_path)
    content_X, contents_Y = get_contents(content_img, output_shape, ctx)
    if style_save_path:
        styles_npy = os.path.join(style_save_path, "styles.npy")
        if os.path.exists(styles_npy):
            styles_Y = nd.load(styles_npy)
        else:
            _, styles_Y = get_styles(style_img, output_shape, ctx)
            nd.save(styles_npy, styles_Y)
    else:
        _, styles_Y = get_styles(style_img, output_shape, ctx)
    return train(content_X, contents_Y, styles_Y, ctx, lr, max_epochs, lr_decay_epoch)


out_img_shape = out_img_shape.split('*')
output_shape = (int(out_img_shape[0]), int(out_img_shape[1]))
output = process(content_path, style_path, output_shape, style_save_path)

d2l.plt.imsave(out_path, post_process(output).asnumpy())
