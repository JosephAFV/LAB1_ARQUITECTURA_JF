import io, json, sys, unittest, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from app import create_app

class WeatherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.app=create_app({'TESTING':True,'MODEL_PATH':Path(self.tmp.name)/'model.joblib'})
        self.client=self.app.test_client()
    def tearDown(self): self.tmp.cleanup()
    def train(self, data=None, filename='dataset.csv'):
        return self.client.post('/train',data={'dataset':(io.BytesIO(data if data is not None else Path('dataset.csv').read_bytes()),filename)})
    def sample(self): return {k:str(v) for k,v in pd.read_csv('dataset.csv').drop(columns='rain').iloc[0].items()}
    def test_01_initial(self): self.assertEqual(self.client.get('/').status_code,200)
    def test_02_training(self):
        r=self.train(); self.assertEqual(r.status_code,200); self.assertIn(b'Predecir lluvia',r.data)
    def test_03_prediction(self):
        self.train(); r=self.client.post('/predict',data=self.sample()); self.assertEqual(r.status_code,200); self.assertIn(b'Resultado:',r.data)
    def test_04_missing_model(self): self.assertEqual(self.client.post('/predict',data=self.sample()).status_code,400)
    def test_05_missing_file(self): self.assertEqual(self.client.post('/train').status_code,400)
    def test_06_extension(self): self.assertEqual(self.train(b'a,b\n1,2','datos.txt').status_code,400)
    def test_07_missing_target(self): self.assertEqual(self.train(b'a,b\n1,2\n3,4').status_code,400)
    def test_08_single_class(self): self.assertEqual(self.train(b'a,rain\n1,Yes\n2,Yes').status_code,400)
    def test_09_invalid_number(self):
        self.train(); data=self.sample(); data['temperature']='abc'
        r=self.client.post('/predict',data=data); self.assertEqual(r.status_code,400); self.assertIn(b'debe ser num',r.data)
    def test_10_required(self):
        self.train(); data=self.sample(); data['temperature']=''
        self.assertEqual(self.client.post('/predict',data=data).status_code,400)
    def test_11_non_finite(self):
        self.train(); data=self.sample(); data['temperature']='NaN'
        self.assertEqual(self.client.post('/predict',data=data).status_code,400)
    def test_12_reload(self):
        self.train(); r=self.client.get('/'); self.assertIn(b'Predecir lluvia',r.data); self.assertIn(b'Accuracy',r.data)
    def test_16_new_dataset(self):
        self.train(); home=self.client.get('/'); self.assertIn(b'href="/?nuevo=1"', home.data)
        r=self.client.get('/?nuevo=1')
        self.assertIn(b'name="dataset"',r.data)
    def test_13_insufficient(self):
        self.assertEqual(self.train(b'a,rain\n1,Yes\n2,Yes\n3,No\n4,No').status_code,400)
    def test_14_categorical(self):
        data=b'temperature,sky,rain\n10,cloudy,Yes\n11,clear,No\n12,cloudy,Yes\n13,clear,No\n14,cloudy,Yes\n15,clear,No\n16,cloudy,Yes\n17,clear,No\n18,cloudy,Yes\n19,clear,No\n'
        self.assertEqual(self.train(data).status_code,200)
        self.assertEqual(self.client.post('/predict',data={'temperature':'12','sky':'new'}).status_code,200)
    def test_15_failure_preserves_model(self):
        self.train(); self.train(b'a,rain\n1,Yes\n2,Yes')
        self.assertEqual(self.client.post('/predict',data=self.sample()).status_code,200)

if __name__=='__main__': unittest.main(verbosity=2)
