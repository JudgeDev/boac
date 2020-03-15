# standard library imports
from io import BytesIO

# related third party imports
import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

# local imports

templates = Jinja2Templates(directory='templates')

export_file_url = 'https://www.dropbox.com/s/6bgq8t6yextloqp/export.pkl?raw=1'
export_file_name = 'export.pkl'

classes = ['black', 'grizzly', 'teddys']
path = Path(__file__).parent

app = Starlette()
# allow cross-origin requests from browsers
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='static'))  # needed to add path for local env

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    print("Waiting for learner file")
    await download_file(export_file_url, path / export_file_name)
    try:
        print("Trying to load learner")
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
#tasks = [asyncio.ensure_future(setup_learner())]
#learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    template = 'index.html'
    context = {"request": request}
    return templates.TemplateResponse(template, context)

    #html_file = path / 'view' / 'index.html'
    #return HTMLResponse(html_file.open(encoding="utf-8").read())  # need to add encoding for local env


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
