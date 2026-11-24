"""Cheap desktop-side worker supervision; never loads inference libraries."""
import ctypes
import os
import time
from .compute_policy import memory_pressure_message,idle_timeout,GIB


class PerformanceInfo(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),*[(key,ctypes.c_size_t) for key in
        ('CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache',
         'KernelTotal','KernelPaged','KernelNonpaged','PageSize')],
        ('HandleCount',ctypes.c_ulong),('ProcessCount',ctypes.c_ulong),('ThreadCount',ctypes.c_ulong)]


def commit_headroom():
    if os.name!='nt':return None
    info=PerformanceInfo();info.cb=ctypes.sizeof(info)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info),info.cb):return None
    return (info.CommitLimit-info.CommitTotal)*info.PageSize


class ProcessingGuard:
    # paging: low physical RAM is not a stop while Windows can still allocate memory (it moves other programs' memory to
    # the page file); `tight` then says that RAM is under the reserve now. The offline window asks for it, PEAR does not.
    paging=False;tight=False
    peak_ram=peak_commit=0  # the largest working set and private commit the worker tree reached, in bytes

    def __init__(self,pid,now=None,resource_safety=None,paging=False):
        import psutil
        from .resource_safety import enabled
        self.resource_safety=enabled() if resource_safety is None else bool(resource_safety)
        self.paging=bool(paging)
        self.process=psutil.Process(pid)
        self.created=self.process.create_time()
        self.last_output=self.last_cpu_change=time.monotonic() if now is None else now
        self.cpu=0.;self.activity_cpu=0.;self.cpu_samples={};self.low_samples=0
        self.reason=None  # 'memory' or 'stall' once check() has returned a stop message

    def output(self):self.last_output=time.monotonic()

    def check(self,now=None):
        import psutil
        now=time.monotonic() if now is None else now
        try:
            if self.process.create_time()!=self.created:return None
            processes=[self.process,*self.process.children(recursive=True)]
            samples={};ram=commit_used=0
            for process in processes:
                try:
                    used=process.memory_info()
                    ram+=used.rss;commit_used+=getattr(used,'pagefile',0)
                except (psutil.Error,OSError,AttributeError):pass
                try:
                    value=process.cpu_times()
                    key=(process.pid,process.create_time())
                    value=value.user+value.system
                    samples[key]=value
                    self.cpu+=max(0.,value-self.cpu_samples.get(key,0.))
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            self.cpu_samples=samples
            self.peak_ram=max(self.peak_ram,ram);self.peak_commit=max(self.peak_commit,commit_used)
            if self.cpu>self.activity_cpu+.02:
                self.last_cpu_change=now
                self.activity_cpu=self.cpu
        except psutil.NoSuchProcess:return None
        except psutil.AccessDenied:pass
        if self.resource_safety:
            memory=psutil.virtual_memory();commit=commit_headroom()
            message=memory_pressure_message(memory.available,memory.total,commit)
            critical=memory.available<256*1024**2 or (commit is not None and commit<256*1024**2)
            if self.paging and commit is not None:
                # Measured: with headroom left, Windows pages other programs out and the job finishes, only slower.
                self.tight=message is not None and commit>=GIB
                if self.tight:message=None
                critical=commit<256*1024**2
            low=message is not None
            self.low_samples=self.low_samples+1 if low else 0
            if self.low_samples>=2 or critical:
                self.reason='memory'
                return message
        if idle_timeout(self.last_output,self.last_cpu_change,now):
            self.reason='stall'
            return 'Processing stopped after 180 seconds without output or CPU activity. The app is still available; retry or use a shorter clip.'
        return None
