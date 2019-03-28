#pragma once
#include <windows.h>
#include <vector>
#include <functional>
#include "Debug.h"

namespace AaltoGames
{
	template <class WorkloadDataType> class WorkerThreadManager
	{
	public:
		class WorkerThread
		{
		public:
			static DWORD WINAPI threadProc(LPVOID lpParameter)
			{
				WorkerThread *t=(WorkerThread *)lpParameter;
				while (!t->terminate)
				{
					WaitForSingleObject(t->waitEvent,INFINITE);
					if (t->workload)
						t->workload(t->data);
					t->workload=nullptr;
					SetEvent(t->doneEvent);
				}
				t->terminated=true;
				return 0;
			}

			HANDLE threadHandle;
			DWORD threadId;
			HANDLE waitEvent,doneEvent;
			std::function<void(WorkloadDataType)> workload;
//			volatile void(*workload)(WorkloadDataType); 
			volatile WorkloadDataType data;
			volatile bool terminate;
			volatile bool terminated;
			WorkerThread(HANDLE doneEvent)
				:terminate(false),terminated(false),workload(nullptr),data(0),doneEvent(doneEvent)
			{
				waitEvent=CreateEvent(NULL,false,false,NULL);  //autoreset
				threadHandle = CreateThread( 
					NULL,                   // default security attributes
					0,                      // use default stack size  
					threadProc,       // thread function name
					this,          // argument to thread function 
					0,                      // use default creation flags 
					&threadId);   // returns the thread identifier 

			}
			~WorkerThread()
			{
				terminate=true;
				workload=nullptr;
				SetEvent(waitEvent);
				while (!terminated)
					Sleep(1);
			}
			//void process(std::function<void(WorkloadDataType)> workload, WorkloadDataType data)
			//{
			//	this->workload=workload;
			//	this->data=data;
			//	SetEvent(waitEvent);
			//}
			void wait()
			{
				WaitForSingleObject(doneEvent,INFINITE);
			}
		};
		std::vector<WorkerThread *> threads;
		HANDLE doneEvent;
		WorkerThreadManager(int nThreads)
		{
			doneEvent=CreateEvent(NULL,false,false,NULL);  //autoreset
			for (int i = 0; i < nThreads; i++)
			{
				threads.push_back(new WorkerThread(doneEvent));
			}
		}
		int nThreadsBusy()
		{
			int result = 0;
			for (WorkerThread *w : threads)
			{
				if (w->workload != nullptr)
					result++;
			}
			return result;
		}

        bool threadsAvailable()
        {
            return nThreadsBusy() != (int)threads.size();
        }
        //This may block until a thread becomes available.
        //Returns the index of the thread, useful if the caller has allocated thread-specific resources
        int allocateThread()
        {
            while (!threadsAvailable())
                WaitForSingleObject(doneEvent,INFINITE);
            for (size_t i = 0; i < threads.size(); i++)
            {
                if (threads[i]->workload == NULL)
                    return i;
            }
			Debug::throwError("WorkerThreadManager.allocateThread() failed.");
			return -1;
        }
        void submitJob(std::function<void(WorkloadDataType)> workload, WorkloadDataType data, int threadIdx = -1)
        //void submitJob(void(*workload)(WorkloadDataType), WorkloadDataType data, int threadIdx = -1)
        {

            if (threadIdx < 0)
                threadIdx = allocateThread();
			threads[threadIdx]->workload=workload;
            //threads[threadIdx]->workload = (volatile void(*)(WorkloadDataType))workload;
			threads[threadIdx]->data=data;
 			SetEvent(threads[threadIdx]->waitEvent);
        }
        void waitAll()
        {
			while (nThreadsBusy() > 0)
			{
				if (WaitForSingleObject(doneEvent,20000)!=WAIT_OBJECT_0)
				{
					Debug::throwError("Waiting for threads failed.");
					break;
				}
			}
        }
        ~WorkerThreadManager()
        {
			waitAll();
			for (WorkerThread *w : threads)
			{
                w->workload = nullptr;
				w->terminate=true;
				SetEvent(w->waitEvent);
            }
			for (WorkerThread *w : threads)
			{
				while (!w->terminated)
					Sleep(1);
            }
        }
    };
} //namespace AaltoGames