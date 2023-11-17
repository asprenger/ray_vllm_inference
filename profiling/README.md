# Profiling

## Profile vLLM using NVIDIA Nsight profiler

[NVIDIA Nsight](https://developer.nvidia.com/nsight-systems) is a profiler tool designed to visualize an application’s 
performance on CPUs and GPUs. See [Get Started With Nsight Systems](https://developer.nvidia.com/nsight-systems/get-started) 
on how to install and start using NSight.

### Code usage

Here is a basic template how to use the Nsight API for a PyTorch model:

    torch.cuda.cudart().cudaProfilerStart()

    for i in range(nb_iters):

        # push range for current iteration
        torch.cuda.nvtx.range_push(f"iteration{i}")

        # push range for forward
        torch.cuda.nvtx.range_push("forward")
        output = model(data)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("backward")
        loss.backward()
        torch.cuda.nvtx.range_pop()

        # pop iteration range
        torch.cuda.nvtx.range_pop()

    torch.cuda.cudart().cudaProfilerStop()

Manual torch.cuda.nvtx.range_push/pop calls in your script are very helpful to orient yourself and immediately see where 
your code spends an unexpected amount of time.

### Profiling

Profile vLLM with a LLaMA-2-7b model:

    nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s process-tree --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true -f true -o profile_report_cpu python profile_vllm_nsight.py --model="meta-llama/Llama-2-7b-chat-hf" --batch-size 8 --profile

This will create a report file `profile_report_cpu.nsys-rep` that can be analysed in the NSight UI.

Arguments:

 * -t cuda,nvtx,osrt,cudnn,cublas: selects the APIs to be traced
 * --capture-range=cudaProfilerApi: profiling will start when cudaProfilerStart() is invoked 
 * --cudabacktrace=true: When tracing CUDA APIs, enable the collection of a backtrace when a CUDA API is invoked. 
 * -s cpu Sample the cpu stack periodically.  Stack samples show up as little tickmarks on the cpu timeline.
   Last time i checked they were orange, but still easy to miss. Mouse over them to show the backtrace at that point.
   -s cpu can increase cpu overhead substantially (I've seen 2X or more) so be aware of that distortion.
   -s none disables cpu sampling.  Without cpu sampling, the profiling overhead is reduced.
   Use -s none if you want the timeline to better represent a production job (api calls and kernels will
   still appear on the profile, but profiling them doesn't distort the timeline nearly as much).
 * -x: Quit the profiler when the app exits.
 * -w: add process stdin, stdout and console output to the report files
 * -o: Output report filename
 * -f: force overwrite of output files
 * --cudabacktrace-threshold=10000 # Threshold (in nanosec) that determines how long a cuda api call
                                # must run to trigger a backtrace.  10 microsec is a reasonable value
                                # (most kernel launches should take less than 10 microsec) but you
                                # should retune if you see a particular api call you'd like to investigate.
                                # Requires --cudabacktrace=true and -s cpu.
 * --osrt-threshold=10000 # Threshold (in nanosec) that determines how long an os runtime call (eg sleep)
                       # must run to trigger a backtrace.
                       # Backtrace collection for os runtime calls that exceed this threshold should
                       # occur by default if -s cpu is enabled.


CPU sampling (-s) is great for getting backtraces that shows where particular timeline calls originate in the code, but also inflates CPU overhead (sometimes dramatically, 2X or more). So with -s cpu, you shouldn’t expect a realistic view of CPU whitespace.

If the application creates child processes, `nsys`` willprofile those as well. They will show up as separate processes with
separate timelines when you open the report in the Nsight UI.