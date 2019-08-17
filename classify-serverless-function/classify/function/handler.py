import fastai.text as text
import fastai

class ToxicityclassifierConfig():
    model_folder = './models'


class ToxicityCheck:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = text.Path(
            ToxicityclassifierConfig.model_folder)
        text.defaults.device = text.torch.device('cpu')
        fastai.torch_core.defaults.device = text.torch.device('cpu')
        self.learner = text.load_learner(path, 'text_toxicity.pkl').to_fp32()

    def isToxic(self, comment):
        result = self.learner.predict(comment)
        return result[2][1].data.numpy() > result[2][0].data.numpy()


def handle(req):
    """handle a request to the function
    Args:
        req (str): request body
    """
    check = ToxicityCheck()
    return str(check.isToxic(req.comment))
