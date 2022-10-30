from SegNet import SegNet

if __name__ == '__main__':
    model = SegNet()
    model.train_v2(batch_size=1)
    # model.restore(model.config['SAVE_MODEL_DIR'])
    # model.train(batch_size=1, max_steps=100)
    model.save()
