// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "stream_manager.h"
#include "gpgpusim_entrypoint.h"
#include "cuda-sim/cuda-sim.h"
#include "gpgpu-sim/gpu-sim.h"
#include <sys/resource.h>
unsigned CUstream_st::sm_next_stream_uid = 0;

CUstream_st::CUstream_st() 
{
    m_pending = false;
    m_uid = sm_next_stream_uid++;
    pthread_mutex_init(&m_lock,NULL);
}

bool CUstream_st::empty()
{
    pthread_mutex_lock(&m_lock);
    bool empty = m_operations.empty();
    pthread_mutex_unlock(&m_lock);
    return empty;
}

bool CUstream_st::busy()
{
    pthread_mutex_lock(&m_lock);
    bool pending = m_pending;
    pthread_mutex_unlock(&m_lock);
    return pending;
}

void CUstream_st::synchronize() 
{
    // called by host thread
    bool done=false;
    do{
        pthread_mutex_lock(&m_lock);
        done = m_operations.empty();
        pthread_mutex_unlock(&m_lock);
    } while ( !done );
}

void CUstream_st::push( const stream_operation &op )
{
    // called by host thread
    pthread_mutex_lock(&m_lock);
    m_operations.push_back( op );
    pthread_mutex_unlock(&m_lock);
}

void CUstream_st::record_next_done()
{
    // called by gpu thread
    pthread_mutex_lock(&m_lock);
    assert(m_pending);
    m_operations.pop_front();
    m_pending=false;
    pthread_mutex_unlock(&m_lock);
}


stream_operation CUstream_st::next()
{
    // called by gpu thread
    pthread_mutex_lock(&m_lock);
    m_pending = true;
    stream_operation result = m_operations.front();
    pthread_mutex_unlock(&m_lock);
    return result;
}

void CUstream_st::print(FILE *fp)
{
    pthread_mutex_lock(&m_lock);
    fprintf(fp,"GPGPU-Sim API:    stream %u has %zu operations\n", m_uid, m_operations.size() );
    std::list<stream_operation>::iterator i;
    unsigned n=0;
    for( i=m_operations.begin(); i!=m_operations.end(); i++ ) {
        stream_operation &op = *i;
        fprintf(fp,"GPGPU-Sim API:       %u : ", n++);
        op.print(fp);
        fprintf(fp,"\n");
    }
    pthread_mutex_unlock(&m_lock);
}
class KAIN_warp_inst
{
    public:
        warp_inst_t *inst;
        simt_mask_t *thread_done;
        addr_vector_t *next_pc;
        unsigned reconvergence_pc;
    KAIN_warp_inst()
    {
		inst = new warp_inst_t(1);
        thread_done= new simt_mask_t;
        next_pc = new addr_vector_t;
		next_pc->resize(32);
    }
    ~KAIN_warp_inst()
    {
       delete inst;
       delete thread_done;
       delete next_pc;
    }
};


class KAIN_warp_inst **block_warpid[299999][100];
//struct KAIN_Rubbish*Rubbish[Cluster_Thread_Num];

//class KAIN_warp_inst **Rubbish[Cluster_Thread_Num];
//volatile unsigned long long Rubbish_P[Cluster_Thread_Num];
//volatile unsigned long long Rubbish_C[Cluster_Thread_Num];


//class KAIN_warp_inst **Memory;
/*
void *Memory_allocate(void *mm)
{
	while(1)
	{
				unsigned long long index = Memory_P % KAIN_memory_buffer;	
				if(Memory[index] == 0)
				{
					class KAIN_warp_inst *tmp;
					if(Memory_allocate_from_OS == 1)
					{
						tmp = new KAIN_warp_inst[KAIN_instruction_buffer];
					}
					else
					{
						bool Has_memory = false;
						while(1)//Wait until Rubbish has memory 
						{
							for(int j = 0; j < Cluster_Thread_Num; j++)	
							{
								unsigned long long index = Rubbish_C[j] % KAIN_rubbish_buffer;
								if(Rubbish[j][index] != 0)
								{
									tmp = Rubbish[j][index];	
									Rubbish[j][index] = 0;
									Rubbish_C[j]++;
									Has_memory = true;
								}						
							}
							if(Has_memory == true)
							{
								break;	
							}	
							else
							{
								printf("No Memory to allocate from Rubbish memory\n");
								sleep(1);	
							}
						}
					}
					Memory[index] = tmp;	
					Memory_P++;	
				}
			//	printf("Allocate memory, ThreadIDCTA is %d, P is %d\n",i,index);
		}
	
}
*/
/*
void *Rubbish_recycle(void *rubbishID)
{
    int ID = *(int *)rubbishID;
    int aver = Cluster_Thread_Num/KAIN_rubbish_thread;
    int begin = 0 + ID * aver;
    int end = Cluster_Thread_Num - (KAIN_rubbish_thread-ID-1)*aver;

	bool sleep_R = true;
	float hate = 1;
	while(1)
	{
		for(int i = begin; i< end; i++)
		{
			//if(Rubbish_C[i]<=Rubbish_P[i]- 100)
		//	{
			    unsigned long long index = Rubbish_C[i] % KAIN_rubbish_buffer;

				if(Rubbish[i][index] != 0)
				{
					delete[] (class KAIN_warp_inst *)Rubbish[i][index];
					Rubbish[i][index] = 0;
				    Rubbish_C[i]++;
				}
		//	}
		}
	}
}
*/

extern volatile long long Count_Block_P[P_C_size];
extern volatile long long Count_Block_C[P_C_size];
//extern volatile int Can_produce_0[P_C_size];
//extern volatile int Can_produce_1[P_C_size];
//extern volatile int Can_consume_0[P_C_size];
//extern volatile int Can_consume_1[P_C_size];
extern volatile int Current_consume[P_C_size];
//extern volatile int Current_produce[P_C_size];
extern unsigned long long Memory_C;
extern int Finished_on_Sim[299999];
extern int Last_Run_Block[Thread_Num];

void *Producer(void *kernel)
{
    gpgpu_cuda_ptx_sim_main_func((kernel_info_t*)kernel); 
}

struct KAIN_SM_run_CTA_str
{
	    kernel_info_t *kernel;
		    int ID;  
};
extern struct KAIN_SM_run_CTA_str *thread_tmp;

int Process_id;
int Process_count;
int Process_waitpid[10];
pthread_t thread_id_producer;
void stream_operation::do_operation( gpgpu_sim *gpu )
{
    if( is_noop() ) 
        return;

    assert(!m_done && m_stream);
    if(g_debug_execution >= 3)
       printf("GPGPU-Sim API: stream %u performing ", m_stream->get_uid() );
    switch( m_type ) {
    case stream_memcpy_host_to_device:
        if(g_debug_execution >= 3)
            printf("memcpy host-to-device\n");
        gpu->memcpy_to_gpu(m_device_address_dst,m_host_address_src,m_cnt);
        m_stream->record_next_done();
        break;
    case stream_memcpy_device_to_host:
        if(g_debug_execution >= 3)
            printf("memcpy device-to-host\n");
        gpu->memcpy_from_gpu(m_host_address_dst,m_device_address_src,m_cnt);
        m_stream->record_next_done();
        break;
    case stream_memcpy_device_to_device:
        if(g_debug_execution >= 3)
            printf("memcpy device-to-device\n");
        gpu->memcpy_gpu_to_gpu(m_device_address_dst,m_device_address_src,m_cnt); 
        m_stream->record_next_done();
        break;
    case stream_memcpy_to_symbol:
        if(g_debug_execution >= 3)
            printf("memcpy to symbol\n");
        gpgpu_ptx_sim_memcpy_symbol(m_symbol,m_host_address_src,m_cnt,m_offset,1,gpu);
        m_stream->record_next_done();
        break;
    case stream_memcpy_from_symbol:
        if(g_debug_execution >= 3)
            printf("memcpy from symbol\n");
        gpgpu_ptx_sim_memcpy_symbol(m_symbol,m_host_address_dst,m_cnt,m_offset,0,gpu);
        m_stream->record_next_done();
        break;
    case stream_kernel_launch:
        if( gpu->can_start_kernel() ) {
            printf("kernel \'%s\' transfer to GPU hardware scheduler\n", m_kernel->name().c_str() );
            if( m_sim_mode )
                gpgpu_cuda_ptx_sim_main_func( m_kernel );//KAIN:modify by kain to adjust the change of this function
            else
            {
                //KAIN add to decouple
           //     static int number;
           //     assert(number == 0);
	      //  extern class KAIN_warp_inst **block_warpid[4000][100]; 
	   	static int init_block_warpid = 0;
		if(init_block_warpid == 0)
		{
			printf("herhe===============kkkkkkkkkkkkk\n");

			
		    for(int i = 0; i < 299999;i++)
			for(int j = 0; j < 100; j++)
			{
		            block_warpid[i][j] = (new KAIN_warp_inst*[2]);//2 is the buffer size
					for(int m = 0; m < 2; m++)
						block_warpid[i][j][m] = 0;
						//block_warpid[i][j][m] = new KAIN_warp_inst[KAIN_instruction_buffer];
			}
			printf("herhe++++++++++++++++++==kkkkkkkkkkkkk\n");
			

			//init Rubbish recycle
			/*	
			for(int i = 0; i < Cluster_Thread_Num; i++)
			{
				//Rubbish[i] = (struct KAIN_Rubbish*)malloc(KAIN_rubbish_buffer*sizeof(struct KAIN_Rubbish));
				Rubbish[i] =  (new KAIN_warp_inst* [KAIN_rubbish_buffer]);
				Rubbish_P[i] = 0;
				Rubbish_C[i] = 0;
			}
				
			for(int i = 0; i < Thread_Num; i++)
			{
				Memory_P[i] = 0;	
				Memory_C[i] = 0;
				Memory[i] = (new KAIN_warp_inst *[KAIN_memory_buffer]);
				for(int j = 0; j < KAIN_memory_buffer; j++)
				{
					Memory[i][j] = 0;	
				}
			}
			*/

            //////////////////////////KAIN set the high priority of the Memory Allloc recycle 
			/*Move to the init
            pthread_attr_t attr_P_R;
            struct sched_param param_P_R;
            pthread_attr_init(&attr_P_R);
            pthread_attr_setinheritsched (&attr_P_R,PTHREAD_EXPLICIT_SCHED);
            pthread_attr_setschedpolicy (&attr_P_R, SCHED_FIFO);
            param_P_R.sched_priority = 99;
            pthread_attr_setschedparam (&attr_P_R, &param_P_R);
            //////////////
            pthread_t thread_id_memory;
            pthread_create(&thread_id_memory,&attr_P_R,Memory_allocate,0);
			*/
            /////////////////////////////////////////////////////////////////////////////////////////////// 
			extern class KAIN_warp_inst **Memory;

            printf("begin allocate memory\n");
            for(long long i = 0; i < Thread_Num*Warp_PerBlock*2+2; i++) 
            {    
				assert(i < 9999999);
                Memory[i] = new KAIN_warp_inst [KAIN_instruction_buffer];
                printf("allcoat memory %d\n",i);
            }    
     
            printf("after allocate memory\n");



		    init_block_warpid = 1;
		}
				Memory_C  = 0;	
				for(int i = 0; i < 299999; i++)
					Finished_on_Sim[i] = 0;
				for(int i = 0; i < Thread_Num; i++)
					Last_Run_Block[i] = -1; 
                for(int i =0; i < P_C_size; i++)
                {
					//Can_produce_0[i] = 1;
					//Can_produce_1[i] = 1;
					//Can_consume_0[i] = 0;
					//Can_consume_1[i] = 0;
					Current_consume[i] = -1;
					//Current_produce[i] = -1;
                    Count_Block_P[i]  = 0;
                    Count_Block_C[i]  = 0;
                }
				


                class function_info *mm = m_kernel->entry();
                kernel_info_t *KAIN_kernel = new kernel_info_t(m_kernel);
		m_kernel->KAIN_set_Mem2_NULL();
                //////////////////////////KAIN set the low priority of the Producer
                pthread_attr_t attr_P;
                struct sched_param param_P;
                pthread_attr_init(&attr_P);
                pthread_attr_setinheritsched (&attr_P,PTHREAD_EXPLICIT_SCHED);
                pthread_attr_setschedpolicy (&attr_P, SCHED_FIFO);
                param_P.sched_priority = 1;
                pthread_attr_setschedparam (&attr_P, &param_P);
                //////////////


				static int fork_process  = 0;
				if(fork_process == 0)
				{
					extern int KAIN_pthread_init;
					KAIN_pthread_init = 0;
					Process_id = 0;
					//fork_process = 1;	
					if(m_kernel->num_blocks()> 1000)// fork process only the blocks > 1000
					{
						Process_count = 2;	
					}
					else
					{
						Process_count = 1;	
					}
					for(int i = 0; i < Process_count-1; i++)
					{
						int id = fork();	
						if(id == 0)
						{
							Process_id = i+1;	
							break;
						}
						else
						{
							//struct rlimit rl;
							//getrlimit(RLIMIT_NPROC,&rl);
							//printf("the manx limite is %d\n",rl);
							Process_waitpid[i] = id;	
						}
					}
				}


                //pthread_create(&thread_id,&attr_P,Producer,KAIN_kernel); 
				KAIN_kernel->KAIN_set_cta(Process_id);
				//Producer(KAIN_kernel);
                pthread_create(&thread_id_producer,&attr_P,Producer,KAIN_kernel); 

				//for(int i = 0; i < Thread_Num; i++)
				//{
				//	thread_tmp[i].kernel = KAIN_kernel;
				//}



				//////////////////set the CPU affinity
				//////////////////
                //gpgpu_cuda_ptx_sim_main_func( m_kernel );//KAIN:modify by kain to adjust the change of this function
               // pthread_join(thread_id,NULL);
                //KAIN add to delete the dependency between functional and performance simulationg
               //	sleep(2400); 
			   //sleep(4);
				printf("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n");
				fflush(stdout);
                m_kernel->KAIN_clear_cta(mm);
				m_kernel->KAIN_set_cta(Process_id);

                gpu->launch( m_kernel );
          //      number++;
            }
        }
        break;
    case stream_event: {
        printf("event update\n");
        time_t wallclock = time((time_t *)NULL);
        m_event->update( gpu_tot_sim_cycle, wallclock );
        m_stream->record_next_done();
        } 
        break;
    default:
        abort();
    }
    m_done=true;
  //  fflush(stdout);
}

void stream_operation::print( FILE *fp ) const
{
    fprintf(fp," stream operation " );
    switch( m_type ) {
    case stream_event: fprintf(fp,"event"); break;
    case stream_kernel_launch: fprintf(fp,"kernel"); break;
    case stream_memcpy_device_to_device: fprintf(fp,"memcpy device-to-device"); break;
    case stream_memcpy_device_to_host: fprintf(fp,"memcpy device-to-host"); break;
    case stream_memcpy_host_to_device: fprintf(fp,"memcpy host-to-device"); break;
    case stream_memcpy_to_symbol: fprintf(fp,"memcpy to symbol"); break;
    case stream_memcpy_from_symbol: fprintf(fp,"memcpy from symbol"); break;
    case stream_no_op: fprintf(fp,"no-op"); break;
    }
}

stream_manager::stream_manager( gpgpu_sim *gpu, bool cuda_launch_blocking ) 
{
    m_gpu = gpu;
    m_service_stream_zero = false;
    m_cuda_launch_blocking = cuda_launch_blocking;
    pthread_mutex_init(&m_lock,NULL);
}

void stream_manager::register_finished_kernel( unsigned grid_uid ) 
{
    // called by gpu simulation thread
    pthread_mutex_lock(&m_lock);
    CUstream_st *stream = m_grid_id_to_stream[grid_uid];
    kernel_info_t *kernel = stream->front().get_kernel();
    assert( grid_uid == kernel->get_uid() );
    stream->record_next_done();
    m_grid_id_to_stream.erase(grid_uid);
    delete kernel;
    pthread_mutex_unlock(&m_lock);
}

stream_operation stream_manager::front() 
{
    // called by gpu simulation thread
    stream_operation result;
    pthread_mutex_lock(&m_lock);
    if( concurrent_streams_empty() )
        m_service_stream_zero = true;
    if( m_service_stream_zero ) {
        if( !m_stream_zero.empty() ) {
            if( !m_stream_zero.busy() ) {
                result = m_stream_zero.next();
                if( result.is_kernel() ) {
                    unsigned grid_id = result.get_kernel()->get_uid();
                    m_grid_id_to_stream[grid_id] = &m_stream_zero;
                }
            }
        } else {
            m_service_stream_zero = false;
        }
    } else {
        std::list<struct CUstream_st*>::iterator s;
        for( s=m_streams.begin(); s != m_streams.end(); s++) {
            CUstream_st *stream = *s;
            if( !stream->busy() && !stream->empty() ) {
                result = stream->next();
                if( result.is_kernel() ) {
                    unsigned grid_id = result.get_kernel()->get_uid();
                    m_grid_id_to_stream[grid_id] = stream;
                }
                break;
            }
        }
    }
    pthread_mutex_unlock(&m_lock);
    return result;
}

void stream_manager::add_stream( struct CUstream_st *stream )
{
    // called by host thread
    pthread_mutex_lock(&m_lock);
    m_streams.push_back(stream);
    pthread_mutex_unlock(&m_lock);
}

void stream_manager::destroy_stream( CUstream_st *stream )
{
    // called by host thread
    pthread_mutex_lock(&m_lock);
    while( !stream->empty() )
        ; 
    std::list<CUstream_st *>::iterator s;
    for( s=m_streams.begin(); s != m_streams.end(); s++ ) {
        if( *s == stream ) {
            m_streams.erase(s);
            break;
        }
    }
    delete stream; 
    pthread_mutex_unlock(&m_lock);
}

bool stream_manager::concurrent_streams_empty()
{
    bool result = true;
    // called by gpu simulation thread
    std::list<struct CUstream_st *>::iterator s;
    for( s=m_streams.begin(); s!=m_streams.end();++s ) {
        struct CUstream_st *stream = *s;
        if( !stream->empty() ) {
            //stream->print(stdout);
            result = false;
        }
    }
    return result;
}

bool stream_manager::empty()
{
    bool result = true;
    pthread_mutex_lock(&m_lock);
    if( !concurrent_streams_empty() ) 
        result = false;
    if( !m_stream_zero.empty() ) 
        result = false;
    pthread_mutex_unlock(&m_lock);
    return result;
}

void stream_manager::print( FILE *fp)
{
    pthread_mutex_lock(&m_lock);
    print_impl(fp);
    pthread_mutex_unlock(&m_lock);
}

void stream_manager::print_impl( FILE *fp)
{
    fprintf(fp,"GPGPU-Sim API: Stream Manager State\n");
    std::list<struct CUstream_st *>::iterator s;
    for( s=m_streams.begin(); s!=m_streams.end();++s ) {
        struct CUstream_st *stream = *s;
        if( !stream->empty() ) 
            stream->print(fp);
    }
    if( !m_stream_zero.empty() ) 
        m_stream_zero.print(fp);
}

void stream_manager::push( stream_operation op )
{
    struct CUstream_st *stream = op.get_stream();

    // block if stream 0 (or concurrency disabled) and pending concurrent operations exist
    bool block= !stream || m_cuda_launch_blocking;
    while(block) {
        pthread_mutex_lock(&m_lock);
        block = !concurrent_streams_empty();
        pthread_mutex_unlock(&m_lock);
    };

    pthread_mutex_lock(&m_lock);
    if( stream && !m_cuda_launch_blocking ) {
        stream->push(op);
    } else {
        op.set_stream(&m_stream_zero);
        m_stream_zero.push(op);
    }
    if(g_debug_execution >= 3)
       print_impl(stdout);
    pthread_mutex_unlock(&m_lock);
    if( m_cuda_launch_blocking || stream == NULL ) {
        unsigned int wait_amount = 100; 
        unsigned int wait_cap = 100000; // 100ms 
        while( !empty() ) {
            // sleep to prevent CPU hog by empty spin
            // sleep time increased exponentially ensure fast response when needed 
            usleep(wait_amount); 
            wait_amount *= 2; 
            if (wait_amount > wait_cap) 
               wait_amount = wait_cap; 
        }
    }
}

