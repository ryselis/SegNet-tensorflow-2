from datetime import timedelta

from SegNet import SegNet, SegNetCompatModel

if __name__ == '__main__':
    # model = SegNetCompatModel()
    # model.train_v2(batch_size=1)
    # model.restore(model.config['SAVE_MODEL_DIR'])
    # model.train(batch_size=1, max_steps=100)
    # model.save()
    model = SegNet()
    model.restore()
    model.train(batch_size=2, max_steps=50000, train_duration=timedelta(hours=3.3))
    model.save()
