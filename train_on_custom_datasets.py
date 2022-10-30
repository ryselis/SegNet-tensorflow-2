from SegNet import SegNet

if __name__ == '__main__':
    model = SegNet()
    model.train(batch_size=1)
    model.save()
