from toolbox import update_ui, get_conf
from toolbox import CatchException, report_execption, write_results_to_file
from crazy_functions.crazy_utils import request_gpt_model_in_new_thread_with_ui_alive
from multiprocessing import Process, Pipe
import threading, time
import numpy as np

def take_audio_sentence_flagment(captured_audio):
    """
    判断音频是否到达句尾，如果到了，截取片段
    """
    ready_part = None
    other_part = captured_audio
    return ready_part, other_part


def save_to_wave(numpy_array, sample_rate, filename):
    from scipy.io import wavfile
    # Scale the audio data
    # scaled_data = numpy_array.astype(np.float32)
    # scaled_data /= np.max(np.abs(scaled_data))
    # Save as a wave file
    wavfile.write(filename, sample_rate, numpy_array)


class WhisperProcess(Process):
    def __init__(self, ):
        super().__init__(daemon=True)
        self.parent, self.child = Pipe()

    def run(self):
        import whisper
        import uuid
        import os
        import tempfile
        device, = get_conf('LOCAL_MODEL_DEVICE')
        # tiny small base medium
        model = whisper.load_model("base", device=device)
        temp_folder = tempfile.gettempdir()
        while True:
            sample_rate = 48000
            audio_numpy = self.child.recv()
            print('======== get')
            if len(audio_numpy) < sample_rate:
                self.child.send("")
                time.sleep(1)
                continue
            print('======== genfile')
            filename = f'{temp_folder}/{uuid.uuid4().hex}.wav'
            save_to_wave(audio_numpy, sample_rate, filename)
            print('======== transcribe')
            result = model.transcribe(filename, language='Chinese', verbose=True)
            os.remove(filename)
            sentences = [r['text'] for r in result['segments']]
            sentence = " ".join(sentences)
            print('======== transcribedone')
            self.child.send(sentence)


class InterviewAssistent():
    def __init__(self):
        self.capture_interval = 0.3 # second
        self.keep_latest_n_second = 5
        self.stop = False
        pass

    def init(self, chatbot):
        # 初始化音频采集线程
        self.captured_audio = np.array([])
        self.captured_text = ""
        self.ready_audio_flagment = None
        self.stop = False
        process = WhisperProcess()
        self.parent_pipe = process.parent
        process.start()
        th1 = threading.Thread(target=self.audio_capture_thread, args=(chatbot._cookies['uuid'],))
        th1.daemon = True
        th1.start()
        th2 = threading.Thread(target=self.audio2txt_thread, args=(chatbot._cookies['uuid'],))
        th2.daemon = True
        th2.start()


    def audio_capture_thread(self, uuid):
        # 在一个异步线程中采集音频
        from .live_audio.audio_io import RealtimeAudioDistribution
        rad = RealtimeAudioDistribution()
        while not self.stop:
            time.sleep(self.capture_interval)
            reading = rad.read(uuid.hex)
            if (reading is not None) and len(reading)>0: pass
            else: continue
            self.captured_audio = np.concatenate((self.captured_audio, reading))
            if len(self.captured_audio) > self.keep_latest_n_second * rad.rate:
                self.captured_audio = self.captured_audio[-self.keep_latest_n_second * rad.rate:]

    def audio2txt_thread(self, llm_kwargs):
        # 在一个异步进程中音频转文字
        while not self.stop:
            self.parent_pipe.send(self.captured_audio)
            self.captured_text = self.parent_pipe.recv()
            if len(self.captured_audio) > 0:
                print('音频峰值：', self.captured_audio.max(), '\t解析文本：', self.captured_text)

    def gpt_answer(self, chatbot, history, llm_kwargs):
        i_say = inputs_show_user = self.captured_text
        gpt_say = yield from request_gpt_model_in_new_thread_with_ui_alive(
            inputs=i_say, inputs_show_user=inputs_show_user,
            llm_kwargs=llm_kwargs, chatbot=chatbot, history=history,
            sys_prompt="你是求职者，正在参加面试，请回答问题。"
        )
        yield from update_ui(chatbot=chatbot, history=history)  # 刷新界面
        history.extend([i_say, gpt_say])

    def begin(self, llm_kwargs, plugin_kwargs, chatbot, history):
        # 面试插件主函数
        self.init(chatbot)
        text_buffer = ""  # 用于监控语音捕获文本的变化
        chatbot.append(["", ""])
        while True:
            time.sleep(self.capture_interval//2)
            if text_buffer != self.captured_text:
                i_say = text_buffer = self.captured_text
                chatbot[-1][0] = i_say
                yield from update_ui(chatbot=chatbot, history=history)  # 刷新界面
                gpt_say = yield from self.gpt_answer(chatbot, history, llm_kwargs)
                chatbot[-1][1] = gpt_say
                yield from update_ui(chatbot=chatbot, history=history)  # 刷新界面

@CatchException
def 辅助面试(txt, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, web_port):
    # pip install -U openai-whisper
    chatbot.append(["函数插件功能：辅助面试。请注意，此插件只能同时服务于一人。", "正在预热本地音频转文字模型 ..."])
    yield from update_ui(chatbot=chatbot, history=history) # 刷新界面

    # import whisper
    # device, = get_conf('LOCAL_MODEL_DEVICE')
    # model = whisper.load_model("base", device=device)
    # result = model.transcribe("private_upload/Akie秋绘-未来轮廓.mp3", language='Chinese')
    chatbot.append(["预热本地音频转文字模型完成", "辅助面试助手, 正在监听音频 ..."])
    yield from update_ui(chatbot=chatbot, history=history) # 刷新界面
    ia = InterviewAssistent()
    yield from ia.begin(llm_kwargs, plugin_kwargs, chatbot, history)

