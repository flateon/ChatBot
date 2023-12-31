import os
import pyaudio
import wave

from KWS import KeywordSpotter

def record(folder="temp", fname="temp.wav"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    input('\tPress enter to record : ')

    chunk = 1000 # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16 # 16 bits per sample
    channels = 1 
    fs = 16000 # Record at 44100 samples per second 
    seconds = 2
    filename = fname

    p = pyaudio.PyAudio() # Create an interface to PortAudio

    print('\tRECORDING.....', end=' ', flush=True)
    # p.going() = True

    stream = p.open(format = sample_format,
                    channels =channels,
                    rate = fs,
                    frames_per_buffer = chunk,
                    input = True)
                    # input_device_index = 1)
    frames = [] #initialize array to store frames 

    # Store data in chunks for 3 seconds 
    for i in range(0, int(fs / chunk * seconds)) : 
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream 
    stream.stop_stream()
    stream.close()
    #terminate the PortAudio interface 
    p.terminate()

    print('FINISHED')

    # Save record dato as a wave file 
    wf = wave.open(os.path.join(folder, filename),'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()


if __name__ == '__main__':
    name = input("Recording Name: ")
    
    doTemplate = input("Record templates? [y]/n: ")
    if not doTemplate.lower() == 'n':
        template_num = int(input("\tNumber of templates to record: "))
        for i in range(1, template_num+1):
            record(f"./Records/{name}", f"template{i}.wav")
    
    doExample = input("Record examples? [y]/n: ")
    if not doExample.lower() == 'n':
        examples = ['positive', 'negative', 'noise']
        for i in range(len(examples)):
            print(f"\tRecord {examples[i]} example", end=' ', flush=True)
            record(f"./Records/{name}", f"{examples[i]}.wav")
    
    doTest = input("Do test? y/[n]: ")
    if doTest.lower() == 'y':
        spotter = KeywordSpotter(f"./Records/{name}")
        spotter.test()
        
        doSingleTest = input("Do single test? y/[n]: ")
        while doSingleTest.lower() == 'y':
            record(f"./Records/{name}", f"test.wav")
            spotter.single_test(f"./Records/{name}/test.wav")
            doQuit = input("\tQuit single test? y/[n]: ")
            if doQuit.lower() == 'y':
                break
