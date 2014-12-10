// Copyright (c) 2009-2011, Tor M. Aamodt, Inderpreet Singh,
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

#ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED
#define ABSTRACT_HARDWARE_MODEL_INCLUDED

enum _memory_space_t {
   undefined_space=0,
   reg_space,
   local_space,
   shared_space,
   param_space_unclassified,
   param_space_kernel,  /* global to all threads in a kernel : read-only */
   param_space_local,   /* local to a thread : read-writable */
   const_space,
   tex_space,
   surf_space,
   global_space,
   generic_space,
   instruction_space
};

#ifdef __cplusplus

#include <string.h>
#include <stdio.h>

typedef unsigned long long new_addr_type;
typedef unsigned address_type;
typedef unsigned addr_t;

// the following are operations the timing model can see 

enum uarch_op_t {
   NO_OP=-1,
   ALU_OP=1,
   SFU_OP,
   ALU_SFU_OP,
   LOAD_OP,
   STORE_OP,
   BRANCH_OP,
   BARRIER_OP,
   MEMORY_BARRIER_OP
};
typedef enum uarch_op_t op_type;

enum _memory_op_t {
	no_memory_op = 0,
	memory_load,
	memory_store
};

#include <bitset>
#include <list>
#include <vector>
#include <assert.h>
#include <stdlib.h>
#include <map>

#if !defined(__VECTOR_TYPES_H__)
struct dim3 {
   unsigned int x, y, z;
};
#endif

#if 0

// detect gcc 4.3 and use unordered map (part of c++0x)
// unordered map doesn't play nice with _GLIBCXX_DEBUG, just use a map if its enabled.
#if  defined( __GNUC__ ) and not defined( _GLIBCXX_DEBUG )
#if __GNUC__ >= 4 && __GNUC_MINOR__ >= 3
   #include <unordered_map>
   #define my_hash_map std::unordered_map
#else
   #include <ext/hash_map>
   namespace std {
      using namespace __gnu_cxx;
   }
   #define my_hash_map std::hash_map
#endif
#else
   #include <map>
   #define my_hash_map std::map
   #define USE_MAP
#endif

#endif
struct KAIN_Rubbish
{
	unsigned Block;
	unsigned warp;
	long long index; 
};
#define Thread_Num 300//240//128//128 //MAX is 8*30 Add by KAIN: thread number to simualte run cta
#define Warp_PerBlock 16//16//16for the Merge_sort//8 //Warp count per block
//#define Process_count 2
#define AllCores 30 // clusters * cores/cluster


#define Cluster_Thread_Num 10//10 
#define P_C_size 13335792// we use  BlockID * KAIN_Warp_counts + WarpID; to increae P C number 
#define KAIN_Warp_counts 33 
#define KAIN_instruction_buffer 1500//2500 


#define KAIN_rubbish_buffer 200000
#define KAIN_memory_buffer 10	
#define KAIN_rubbish_thread 1 //It  must be dived by Cluster Thread NUM!!



//block_warpid[4000][100]
#include <sched.h>
void increment_x_then_y_then_z( dim3 &i, const dim3 &bound);


extern int End_Block_Process[5];
extern int Begin_Block_Process[5];

extern pthread_mutex_t kernel_cores_running;
class kernel_info_t {
public:
//   kernel_info_t()
//   {
//      m_valid=false;
//      m_kernel_entry=NULL;
//      m_uid=0;
//      m_num_cores_running=0;
//      m_param_mem=NULL;
//   }
   kernel_info_t( dim3 gridDim, dim3 blockDim, class function_info *entry );
   kernel_info_t(class kernel_info_t *m);
   ~kernel_info_t();

   void KAIN_set_Mem2_NULL(void)
   {
  	m_param_mem = NULL;	 
   }

   void inc_running() { 
	   pthread_mutex_lock(&kernel_cores_running);
	   m_num_cores_running++;
	   pthread_mutex_unlock(&kernel_cores_running);
   }
   void dec_running()
   {
	   pthread_mutex_lock(&kernel_cores_running);
       assert( m_num_cores_running > 0 );
       m_num_cores_running--; 
	   pthread_mutex_unlock(&kernel_cores_running);
   }
   bool running() const { return m_num_cores_running>0; }
   bool done() const 
   {
       return no_more_ctas_to_run() && !running();
   }
   class function_info *entry() { return m_kernel_entry; }
   const class function_info *entry() const { return m_kernel_entry; }

   size_t num_blocks() const
   {
      return m_grid_dim.x * m_grid_dim.y * m_grid_dim.z;
   }
   size_t num_blocks_runID_KAIN(int Thread_ID) const
   {
	 return  m_next_cta[Thread_ID].x + m_grid_dim.x*m_next_cta[Thread_ID].y+m_grid_dim.x*m_grid_dim.y*m_next_cta[Thread_ID].z;
   }
/*
   void Cta_ID_Set_KAIN(size_t cat_per, int Thread_ID)
   {
		cta_per_kain[Thread_ID] = cat_per;		
   }
   
   void set_blocks_runID_KAIN(int Thread_ID)
   {
   		size_t Cta_per;
		size_t Cta_max;//The begin  CTA of This ID,Cta_max-1 is the end of  before ID 
		if(num_blocks()%Thread_Num==0)
			Cta_per = num_blocks()/Thread_Num;
		else{
			size_t extra = num_blocks()%Thread_Num;
			size_t num_block_used = num_blocks()-extra;
			if(Thread_ID < extra)
				Cta_per = num_block_used/Thread_Num+1;
			else
				Cta_per = num_block_used/Thread_Num;
		}
		if(Thread_ID != 0)
			Cta_max = m_next_cta[Thread_ID].x + m_grid_dim.x*m_next_cta[Thread_ID].y+m_grid_dim.x*m_grid_dim.y*m_next_cta[Thread_ID].z+Cta_per;
		else
			Cta_max = Cta_per;
	//	printf("ThreadID is %d, Cta_per is %d\n",Thread_ID,Cta_per);
		Cta_ID_Set_KAIN(Cta_per,Thread_ID);
		if(Thread_ID < Thread_Num-1){
			m_next_cta[Thread_ID+1].z = Cta_max/(m_grid_dim.x*m_grid_dim.y);
			m_next_cta[Thread_ID+1].y = (Cta_max-m_next_cta[Thread_ID+1].z*m_grid_dim.x*m_grid_dim.y)/m_grid_dim.x;
			m_next_cta[Thread_ID+1].x = Cta_max-m_next_cta[Thread_ID+1].z*m_grid_dim.x*m_grid_dim.y - m_next_cta[Thread_ID+1].y*m_grid_dim.x;
			m_init_cta[Thread_ID+1] = m_next_cta[Thread_ID+1];
		}
		
   	}
	*/
   size_t threads_per_cta() const
   {
      return m_block_dim.x * m_block_dim.y * m_block_dim.z;
   } 

   dim3 get_grid_dim() const { return m_grid_dim; }
   dim3 get_cta_dim() const { return m_block_dim; }
   void increment_cta_id_performance(int Thread_ID)//if next funciton used in performace simualtino, the m_next_cta would be 0 twice
   {
  //    printf("incraeate m_next_cta_all\n");
      increment_x_then_y_then_z(m_next_cta_all,m_grid_dim); 
      m_next_cta[Thread_ID] = m_next_cta_all;
      m_next_tid[Thread_ID].x=0;
      m_next_tid[Thread_ID].y=0;
      m_next_tid[Thread_ID].z=0;

   }

   void increment_cta_id(int Thread_ID) 
   { 
    //  printf("incraeate m_next_cta_all\n");
	//  fflush(stdout);
      m_next_cta[Thread_ID] = m_next_cta_all;
      increment_x_then_y_then_z(m_next_cta_all,m_grid_dim); 
      m_next_tid[Thread_ID].x=0;
      m_next_tid[Thread_ID].y=0;
      m_next_tid[Thread_ID].z=0;
   }
   dim3 get_next_cta_id(int Thread_ID) const { return m_next_cta[Thread_ID]; }
   bool no_more_ctas_to_run(void) const//it would be clled by every thread,
   {
       //bool ret = (m_next_cta_all.x >= m_grid_dim.x || m_next_cta_all.y >= m_grid_dim.y || m_next_cta_all.z >= m_grid_dim.z );
//       bool ret = ((m_next_cta_all.x >= m_grid_dim_kain.x && m_grid_dim_kain.x != 0)|| (m_next_cta_all.y >= m_grid_dim_kain.y && m_grid_dim_kain.y != 0) || (m_next_cta_all.z >= m_grid_dim_kain.z && m_grid_dim_kain.z != 0));

		extern int Process_id;
	   int blocks_now = m_next_cta_all.x + m_next_cta_all.y * m_grid_dim.x + m_next_cta_all.z * m_grid_dim.x * m_grid_dim.y;
	   int blocks_can = m_grid_dim_kain.x + m_grid_dim_kain.y * m_grid_dim.x + m_grid_dim_kain.z * m_grid_dim.x * m_grid_dim.y;
	   bool ret = blocks_now >= blocks_can;

	   return ret;
   }



   bool KAIN_clear_cta(class function_info *mm)
   {   
        m_next_cta_all.x = 0;
        m_next_cta_all.y = 0;
        m_next_cta_all.z = 0;
        for(int i = 0; i < Thread_Num; i++)
        {   
            m_next_cta[i].x=0;
            m_next_cta[i].y=0;
            m_next_cta[i].z=0;
            m_next_tid[i]=m_next_cta[i];
            m_active_threads[i].clear();
        }  
        //m_next_cta.x = 0;        
        //m_next_cta.y = 0;
        //m_next_cta.z = 0;
        //m_next_tid.x=0;
        //m_next_tid.y=0;
        //m_next_tid.z=0;
        m_num_cores_running = 0;
        m_kernel_entry = mm; 
   }  
   bool KAIN_set_cta(int process_id)
   {
   		extern int Pure_simulation_blockID;
   		extern int Function_over_thread;
		extern int Process_count;

		Function_over_thread = 0;
  		int all_blocks = num_blocks();	 
		int blocks_per_process = all_blocks / Process_count;
		int leave_blocks = all_blocks % Process_count;
		int begin = blocks_per_process * process_id;
		int end = blocks_per_process * (process_id+1);//end no need to - 1, beacause end is also calculated
		
		Begin_Block_Process[process_id] = begin;
//		if(process_id  != 0)
//			begin = begin-60;
		///
//		if(process_id != Process_count - 1)
//			end = end + 60;//240 is the max CTA on cores
		/////
		if(process_id == Process_count - 1)
			end = end+leave_blocks;//for the last one, end does not satify the check in increament performance
		End_Block_Process[process_id] = end;//This is the logic split
	
		if(process_id == 0 && Process_count > 1)
		{
			Pure_simulation_blockID = end;	
		}
		else
		{
			Pure_simulation_blockID = 999999999;// it means the maxiams	
		}


		m_next_cta_all.z = begin / (m_grid_dim.x*m_grid_dim.y);
		m_next_cta_all.y =  (begin - m_next_cta_all.z * m_grid_dim.x*m_grid_dim.y)/ m_grid_dim.x;
		m_next_cta_all.x = begin - m_next_cta_all.z * m_grid_dim.x*m_grid_dim.y -  m_next_cta_all.y * m_grid_dim.x;
		assert(m_next_cta_all.x + m_next_cta_all.y * m_grid_dim.x + m_next_cta_all.z * m_grid_dim.x * m_grid_dim.y == begin);

		 m_next_cta[0] = m_next_cta_all;//set the cycle simualtion first 
//		if(process_id != Process_count - 1)
//		{
			m_grid_dim_kain.z = end / (m_grid_dim.x*m_grid_dim.y);
			m_grid_dim_kain.y =  (end - m_grid_dim_kain.z * m_grid_dim.x*m_grid_dim.y)/ m_grid_dim.x;
			m_grid_dim_kain.x = end - m_grid_dim_kain.z * m_grid_dim.x*m_grid_dim.y -  m_grid_dim_kain.y * m_grid_dim.x;
			assert(m_grid_dim_kain.x + m_grid_dim_kain.y * m_grid_dim.x + m_grid_dim_kain.z * m_grid_dim.x * m_grid_dim.y == end);
//		}
//		else
//		{
//			m_grid_dim_kain = m_grid_dim;	
//		}
		printf("Begin, Process id :%d, x is %d,y is %d, z is %d\n",process_id,m_next_cta_all.x,m_next_cta_all.y,m_next_cta_all.z);
		printf("End, Process id :%d, x is %d,y is %d, z is %d\n",process_id,m_grid_dim_kain.x,m_grid_dim_kain.y,m_grid_dim_kain.z);
		fflush(stdout);
   }
   bool no_more_ctas_to_run_kain(int Thread_ID, int *Check_again) //it would be clled by every thread,
   {
	   extern unsigned long long Memory_C;
	   extern class KAIN_warp_inst **Memory;
	   extern class KAIN_warp_inst **block_warpid[299999][100];
	   extern int Finished_on_Sim[299999];
	   extern int Last_Run_Block[Thread_Num];
	   extern int Process_id;
	   extern int Function_over_thread;

       extern pthread_mutex_t shared_memory_lookup_mutex;
	   pthread_mutex_lock(&shared_memory_lookup_mutex);
       //bool ret = (m_next_cta_all.x >= m_grid_dim_kain.x || m_next_cta_all.y >= m_grid_dim_kain.y || m_next_cta_all.z >= m_grid_dim_kain.z );
	//	bool ret = ((m_next_cta_all.x >= m_grid_dim_kain.x && m_grid_dim_kain.x != 0)|| (m_next_cta_all.y >= m_grid_dim_kain.y && m_grid_dim_kain.y != 0) || (m_next_cta_all.z >= m_grid_dim_kain.z && m_grid_dim_kain.z != 0));
      // m_next_cta[Thread_ID] = m_next_cta_all;
		int blocks_now = m_next_cta_all.x + m_next_cta_all.y * m_grid_dim.x + m_next_cta_all.z * m_grid_dim.x * m_grid_dim.y;
		int blocks_can = m_grid_dim_kain.x + m_grid_dim_kain.y * m_grid_dim.x + m_grid_dim_kain.z * m_grid_dim.x * m_grid_dim.y;
		bool ret = blocks_now >= blocks_can;


       if(ret == false)
	   {
		   if(Last_Run_Block[Thread_ID] == -1)
		   {
			   increment_cta_id(Thread_ID);
			   int blockID= num_blocks_runID_KAIN(Thread_ID);
			   Last_Run_Block[Thread_ID] = blockID;//set the next BlockID
			   int nwarps;
			   if(threads_per_cta()%32==0 )//  = kernel->threads_per_cta()/32 + 1;//32 is warpsize
				   nwarps = threads_per_cta()/32; 
			   else 
				   nwarps = threads_per_cta()/32 + 1; 
			   for(int i = 0; i < nwarps; i++)	
			   {
					block_warpid[blockID][i][0] = Memory[Memory_C];
					Memory_C++;
					block_warpid[blockID][i][1] = Memory[Memory_C];
					Memory_C++;
					assert(Memory_C < Thread_Num*Warp_PerBlock*2+2);
			   }
		   }
		   else//The thread runs Blocck finished, check the Block is also fininshed on the Gpu-sim
		   {
			   int last_block = Last_Run_Block[Thread_ID];
		  		if(Finished_on_Sim[last_block] == 1) 
				{
					   increment_cta_id(Thread_ID);
					   int blockID= num_blocks_runID_KAIN(Thread_ID);
					   Last_Run_Block[Thread_ID] = blockID;//set the next BlockID
					   int nwarps;
					   if(threads_per_cta()%32==0 )//  = kernel->threads_per_cta()/32 + 1;//32 is warpsize
						   nwarps = threads_per_cta()/32; 
					   else 
						   nwarps = threads_per_cta()/32 + 1; 
					   for(int i = 0; i < nwarps; i++)	
					   {
							block_warpid[blockID][i][0] = block_warpid[last_block][i][0] ;
							block_warpid[blockID][i][1] = block_warpid[last_block][i][1];
					   }
				}
				else
				{
					*Check_again = 1;	
				}
		   }
	   }
	/*kain add in 2014.4.11 to make sure the next kernel's simulation is also right
 	so the process0 must simualte the left blocks in pure functional simulation*/
	 if(ret == true && Process_id== 0 )//40 is set by kain, it means only 40 threads can do pure simulation
	 {
			if (blocks_now >= num_blocks()) 
			{
				ret = true;
				Function_over_thread++;
			}
			else
			{
				increment_cta_id(Thread_ID);
				ret = false;
			}
	 }

	   pthread_mutex_unlock(&shared_memory_lookup_mutex);
       return ret;
   }

   void increment_thread_id(int Thread_ID) { increment_x_then_y_then_z(m_next_tid[Thread_ID],m_block_dim); }
   dim3 get_next_thread_id_3d(int Thread_ID) const  { return m_next_tid[Thread_ID]; }
   unsigned get_next_thread_id(int Thread_ID) const 
   { 
      return m_next_tid[Thread_ID].x + m_block_dim.x*m_next_tid[Thread_ID].y + m_block_dim.x*m_block_dim.y*m_next_tid[Thread_ID].z;
   }
   bool more_threads_in_cta(int Thread_ID) const 
   {
      return m_next_tid[Thread_ID].z < m_block_dim.z && m_next_tid[Thread_ID].y < m_block_dim.y && m_next_tid[Thread_ID].x < m_block_dim.x;
   }
   unsigned get_uid() const { return m_uid; }
   std::string name() const;

   std::list<class ptx_thread_info *> &active_threads(int Thread_ID) { return m_active_threads[Thread_ID]; }
   class memory_space *get_param_memory() { 
     // printf("get m_param_mem, function or performanceKAIN\n"); 
       return m_param_mem; }

private:
   kernel_info_t( const kernel_info_t & ); // disable copy constructor
   void operator=( const kernel_info_t & ); // disable copy operator

   class function_info *m_kernel_entry;

   unsigned m_uid;
   static unsigned m_next_uid;
   dim3 m_grid_dim;
   dim3 m_grid_dim_kain;
   dim3 m_block_dim;
   dim3 m_next_cta_all;
   dim3 m_next_cta[Thread_Num];
   dim3 m_next_tid[Thread_Num];
   dim3 m_init_cta[Thread_Num];
   size_t cta_per_kain[Thread_Num];
   unsigned m_num_cores_running;

   std::list<class ptx_thread_info *> m_active_threads[299999];
   class memory_space *m_param_mem;
};

struct core_config {
    core_config() 
    { 
        m_valid = false; 
        num_shmem_bank=16; 
    }
    virtual void init() = 0;

    bool m_valid;
    unsigned warp_size;

    // off-chip memory request architecture parameters
    int gpgpu_coalesce_arch;

    // shared memory bank conflict checking parameters
    static const address_type WORD_SIZE=4;
    unsigned num_shmem_bank;
    unsigned shmem_bank_func(address_type addr) const
    {
        return ((addr/WORD_SIZE) % num_shmem_bank);
    }
    unsigned mem_warp_parts;  
    unsigned gpgpu_shmem_size;

    // texture and constant cache line sizes (used to determine number of memory accesses)
    unsigned gpgpu_cache_texl1_linesize;
    unsigned gpgpu_cache_constl1_linesize;

	unsigned gpgpu_max_insn_issue_per_warp;
};

// bounded stack that implements simt reconvergence using pdom mechanism from MICRO'07 paper
const unsigned MAX_WARP_SIZE = 32;
typedef std::bitset<MAX_WARP_SIZE> active_mask_t;
#define MAX_WARP_SIZE_SIMT_STACK  MAX_WARP_SIZE
typedef std::bitset<MAX_WARP_SIZE_SIMT_STACK> simt_mask_t;
typedef std::vector<address_type> addr_vector_t;

class simt_stack {
public:
    simt_stack( unsigned wid,  unsigned warpSize);

    void reset();
    void launch( address_type start_pc, const simt_mask_t &active_mask );
    void update( simt_mask_t &thread_done, addr_vector_t &next_pc, address_type recvg_pc );

    const simt_mask_t &get_active_mask() const;
    void     get_pdom_stack_top_info( unsigned *pc, unsigned *rpc ) const;
    unsigned get_rp() const;
    void     print(FILE*fp) const;

protected:
    unsigned m_warp_id;
    unsigned m_stack_top;
    unsigned m_warp_size;
    
    address_type *m_pc;
    simt_mask_t  *m_active_mask;
    address_type *m_recvg_pc;
    unsigned int *m_calldepth;
    
    unsigned long long  *m_branch_div_cycle;
};

#define GLOBAL_HEAP_START 0x80000000
   // start allocating from this address (lower values used for allocating globals in .ptx file)
#define SHARED_MEM_SIZE_MAX (64*1024)
#define LOCAL_MEM_SIZE_MAX (8*1024)
#define MAX_STREAMING_MULTIPROCESSORS 64
#define MAX_THREAD_PER_SM 2048
#define TOTAL_LOCAL_MEM_PER_SM (MAX_THREAD_PER_SM*LOCAL_MEM_SIZE_MAX)
#define TOTAL_SHARED_MEM (MAX_STREAMING_MULTIPROCESSORS*SHARED_MEM_SIZE_MAX)
#define TOTAL_LOCAL_MEM (MAX_STREAMING_MULTIPROCESSORS*MAX_THREAD_PER_SM*LOCAL_MEM_SIZE_MAX)
#define SHARED_GENERIC_START (GLOBAL_HEAP_START-TOTAL_SHARED_MEM)
#define LOCAL_GENERIC_START (SHARED_GENERIC_START-TOTAL_LOCAL_MEM)
#define STATIC_ALLOC_LIMIT (GLOBAL_HEAP_START - (TOTAL_LOCAL_MEM+TOTAL_SHARED_MEM))

#if !defined(__CUDA_RUNTIME_API_H__)

enum cudaChannelFormatKind {
   cudaChannelFormatKindSigned,
   cudaChannelFormatKindUnsigned,
   cudaChannelFormatKindFloat
};

struct cudaChannelFormatDesc {
   int                        x;
   int                        y;
   int                        z;
   int                        w;
   enum cudaChannelFormatKind f;
};

struct cudaArray {
   void *devPtr;
   int devPtr32;
   struct cudaChannelFormatDesc desc;
   int width;
   int height;
   int size; //in bytes
   unsigned dimensions;
};

enum cudaTextureAddressMode {
   cudaAddressModeWrap,
   cudaAddressModeClamp
};

enum cudaTextureFilterMode {
   cudaFilterModePoint,
   cudaFilterModeLinear
};

enum cudaTextureReadMode {
   cudaReadModeElementType,
   cudaReadModeNormalizedFloat
};

struct textureReference {
   int                           normalized;
   enum cudaTextureFilterMode    filterMode;
   enum cudaTextureAddressMode   addressMode[2];
   struct cudaChannelFormatDesc  channelDesc;
};

#endif

class gpgpu_functional_sim_config 
{
public:
    void reg_options(class OptionParser * opp);

    void ptx_set_tex_cache_linesize(unsigned linesize);

    unsigned get_forced_max_capability() const { return m_ptx_force_max_capability; }
    bool convert_to_ptxplus() const { return m_ptx_convert_to_ptxplus; }
    bool use_cuobjdump() const { return m_ptx_use_cuobjdump; }

    int         get_ptx_inst_debug_to_file() const { return g_ptx_inst_debug_to_file; }
    const char* get_ptx_inst_debug_file() const  { return g_ptx_inst_debug_file; }
    int         get_ptx_inst_debug_thread_uid() const { return g_ptx_inst_debug_thread_uid; }
    unsigned    get_texcache_linesize() const { return m_texcache_linesize; }

private:
    // PTX options
    int m_ptx_convert_to_ptxplus;
    int m_ptx_use_cuobjdump;
    unsigned m_ptx_force_max_capability;

    int   g_ptx_inst_debug_to_file;
    char* g_ptx_inst_debug_file;
    int   g_ptx_inst_debug_thread_uid;

    unsigned m_texcache_linesize;
};

class gpgpu_t {
public:
    gpgpu_t( const gpgpu_functional_sim_config &config );
    void* gpu_malloc( size_t size );
    void* gpu_mallocarray( size_t count );
    void  gpu_memset( size_t dst_start_addr, int c, size_t count );
    void  memcpy_to_gpu( size_t dst_start_addr, const void *src, size_t count );
    void  memcpy_from_gpu( void *dst, size_t src_start_addr, size_t count );
    void  memcpy_gpu_to_gpu( size_t dst, size_t src, size_t count );
    
    class memory_space *get_global_memory() { return m_global_mem; }
    class memory_space *get_tex_memory() { return m_tex_mem; }
    class memory_space *get_surf_memory() { return m_surf_mem; }

    void gpgpu_ptx_sim_bindTextureToArray(const struct textureReference* texref, const struct cudaArray* array);
    void gpgpu_ptx_sim_bindNameToTexture(const char* name, const struct textureReference* texref);
    const char* gpgpu_ptx_sim_findNamefromTexture(const struct textureReference* texref);

    const struct textureReference* get_texref(const std::string &texname) const
    {
        std::map<std::string, const struct textureReference*>::const_iterator t=m_NameToTextureRef.find(texname);
        assert( t != m_NameToTextureRef.end() );
        return t->second;
    }
    const struct cudaArray* get_texarray( const struct textureReference *texref ) const
    {
    
	// 	extern pthread_mutex_t shared_memory_lookup_mutex;//ADD by KAIN to see lock
   // 	pthread_mutex_lock(&shared_memory_lookup_mutex);
        std::map<const struct textureReference*,const struct cudaArray*>::const_iterator t=m_TextureRefToCudaArray.find(texref);
        assert(t != m_TextureRefToCudaArray.end());
	//	pthread_mutex_unlock(&shared_memory_lookup_mutex);
        return t->second;
    }
    const struct textureInfo* get_texinfo( const struct textureReference *texref ) const
    {
        std::map<const struct textureReference*, const struct textureInfo*>::const_iterator t=m_TextureRefToTexureInfo.find(texref);
        assert(t != m_TextureRefToTexureInfo.end());
        return t->second;
    }

    const gpgpu_functional_sim_config &get_config() const { return m_function_model_config; }
    FILE* get_ptx_inst_debug_file() { return ptx_inst_debug_file; }

protected:
    const gpgpu_functional_sim_config &m_function_model_config;
    FILE* ptx_inst_debug_file;

    class memory_space *m_global_mem;
    class memory_space *m_tex_mem;
    class memory_space *m_surf_mem;
    
    unsigned long long m_dev_malloc;
    
    std::map<std::string, const struct textureReference*> m_NameToTextureRef;
    std::map<const struct textureReference*,const struct cudaArray*> m_TextureRefToCudaArray;
    std::map<const struct textureReference*, const struct textureInfo*> m_TextureRefToTexureInfo;
};

struct gpgpu_ptx_sim_kernel_info 
{
   // Holds properties of the kernel (Kernel's resource use). 
   // These will be set to zero if a ptxinfo file is not present.
   int lmem;
   int smem;
   int cmem;
   int regs;
   unsigned ptx_version;
   unsigned sm_target;
};

struct gpgpu_ptx_sim_arg {
   gpgpu_ptx_sim_arg() { m_start=NULL; }
   gpgpu_ptx_sim_arg(const void *arg, size_t size, size_t offset)
   {
      m_start=arg;
      m_nbytes=size;
      m_offset=offset;
   }
   const void *m_start;
   size_t m_nbytes;
   size_t m_offset;
};

typedef std::list<gpgpu_ptx_sim_arg> gpgpu_ptx_sim_arg_list_t;

class memory_space_t {
public:
   memory_space_t() { m_type = undefined_space; m_bank=0; }
   memory_space_t( const enum _memory_space_t &from ) { m_type = from; m_bank = 0; }
   bool operator==( const memory_space_t &x ) const { return (m_bank == x.m_bank) && (m_type == x.m_type); }
   bool operator!=( const memory_space_t &x ) const { return !(*this == x); }
   bool operator<( const memory_space_t &x ) const 
   { 
      if(m_type < x.m_type)
         return true;
      else if(m_type > x.m_type)
         return false;
      else if( m_bank < x.m_bank )
         return true; 
      return false;
   }
   enum _memory_space_t get_type() const { return m_type; }
   unsigned get_bank() const { return m_bank; }
   void set_bank( unsigned b ) { m_bank = b; }
   bool is_const() const { return (m_type == const_space) || (m_type == param_space_kernel); }
   bool is_local() const { return (m_type == local_space) || (m_type == param_space_local); }

private:
   enum _memory_space_t m_type;
   unsigned m_bank; // n in ".const[n]"; note .const == .const[0] (see PTX 2.1 manual, sec. 5.1.3)
};

const unsigned MAX_MEMORY_ACCESS_SIZE = 128;
typedef std::bitset<MAX_MEMORY_ACCESS_SIZE> mem_access_byte_mask_t;
#define NO_PARTIAL_WRITE (mem_access_byte_mask_t())

enum mem_access_type {
   GLOBAL_ACC_R, 
   LOCAL_ACC_R, 
   CONST_ACC_R, 
   TEXTURE_ACC_R, 
   GLOBAL_ACC_W, 
   LOCAL_ACC_W,
   L1_WRBK_ACC,
   L2_WRBK_ACC, 
   INST_ACC_R,
   NUM_MEM_ACCESS_TYPE
};

enum cache_operator_type {
    CACHE_UNDEFINED, 

    // loads
    CACHE_ALL,          // .ca
    CACHE_LAST_USE,     // .lu
    CACHE_VOLATILE,     // .cv
                       
    // loads and stores 
    CACHE_STREAMING,    // .cs
    CACHE_GLOBAL,       // .cg

    // stores
    CACHE_WRITE_BACK,   // .wb
    CACHE_WRITE_THROUGH // .wt
};

class mem_access_t {
public:
   mem_access_t() { init(); }
   mem_access_t( mem_access_type type, 
                 address_type address, 
                 unsigned size,
                 bool wr )
   {
       init();
       m_type = type;
       m_addr = address;
       m_req_size = size;
       m_write = wr;
   }
   mem_access_t( mem_access_type type, 
                 address_type address, 
                 unsigned size, 
                 bool wr, 
                 const active_mask_t &active_mask,
                 const mem_access_byte_mask_t &byte_mask )
    : m_warp_mask(active_mask), m_byte_mask(byte_mask)
   {
      init();
      m_type = type;
      m_addr = address;
      m_req_size = size;
      m_write = wr;
   }

   new_addr_type get_addr() const { return m_addr; }
   unsigned get_size() const { return m_req_size; }
   const active_mask_t &get_warp_mask() const { return m_warp_mask; }
   bool is_write() const { return m_write; }
   enum mem_access_type get_type() const { return m_type; }
   mem_access_byte_mask_t get_byte_mask() const { return m_byte_mask; }

   void print(FILE *fp) const
   {
       fprintf(fp,"addr=0x%llx, %s, size=%u, ", m_addr, m_write?"store":"load ", m_req_size );
       switch(m_type) {
       case GLOBAL_ACC_R:  fprintf(fp,"GLOBAL_R"); break;
       case LOCAL_ACC_R:   fprintf(fp,"LOCAL_R "); break;
       case CONST_ACC_R:   fprintf(fp,"CONST   "); break;
       case TEXTURE_ACC_R: fprintf(fp,"TEXTURE "); break;
       case GLOBAL_ACC_W:  fprintf(fp,"GLOBAL_W"); break;
       case LOCAL_ACC_W:   fprintf(fp,"LOCAL_W "); break;
       case L2_WRBK_ACC:   fprintf(fp,"L2_WRBK "); break;
       case INST_ACC_R:    fprintf(fp,"INST    "); break;
       case L1_WRBK_ACC:   fprintf(fp,"L1_WRBK "); break;
       default:            fprintf(fp,"unknown "); break;
       }
   }

private:
   void init() 
   {
      m_uid=++sm_next_access_uid;
      m_addr=0;
      m_req_size=0;
   }

   unsigned      m_uid;
   new_addr_type m_addr;     // request address
   bool          m_write;
   unsigned      m_req_size; // bytes
   mem_access_type m_type;
   active_mask_t m_warp_mask;
   mem_access_byte_mask_t m_byte_mask;

   static unsigned sm_next_access_uid;
};

class mem_fetch;

class mem_fetch_interface {
public:
    virtual bool full( unsigned size, bool write ) const = 0;
    virtual void push( mem_fetch *mf ) = 0;
};

class mem_fetch_allocator {
public:
    virtual mem_fetch *alloc( new_addr_type addr, mem_access_type type, unsigned size, bool wr ) const = 0;
    virtual mem_fetch *alloc( const class warp_inst_t &inst, const mem_access_t &access ) const = 0;
};

// the maximum number of destination, source, or address uarch operands in a instruction
#define MAX_REG_OPERANDS 8

struct dram_callback_t {
   dram_callback_t() { function=NULL; instruction=NULL; thread=NULL; }
   void (*function)(const class inst_t*, class ptx_thread_info*);
   const class inst_t* instruction;
   class ptx_thread_info *thread;
};

class inst_t {
public:
    inst_t()
    {
        m_decoded=false;
        pc=(address_type)-1;
        reconvergence_pc=(address_type)-1;
        op=NO_OP; 
        memset(out, 0, sizeof(unsigned)); 
        memset(in, 0, sizeof(unsigned)); 
        is_vectorin=0; 
        is_vectorout=0;
        space = memory_space_t();
        cache_op = CACHE_UNDEFINED;
        latency = 1;
        initiation_interval = 1;
        for( unsigned i=0; i < MAX_REG_OPERANDS; i++ ) {
            arch_reg.src[i] = -1;
            arch_reg.dst[i] = -1;
        }
        isize=0;
    }
    bool valid() const { return m_decoded; }
    virtual void print_insn( FILE *fp ) const 
    {
        fprintf(fp," [inst @ pc=0x%04x] ", pc );
    }
    bool is_load() const { return (op == LOAD_OP || memory_op == memory_load); }
    bool is_store() const { return (op == STORE_OP || memory_op == memory_store); }

    address_type pc;        // program counter address of instruction
    unsigned isize;         // size of instruction in bytes 
    op_type op;             // opcode (uarch visible)
    _memory_op_t memory_op; // memory_op used by ptxplus 

    address_type reconvergence_pc; // -1 => not a branch, -2 => use function return address
    
    unsigned out[4];
    unsigned in[4];
    unsigned char is_vectorin;
    unsigned char is_vectorout;
    int pred; // predicate register number
    int ar1, ar2;
    // register number for bank conflict evaluation
    struct {
        int dst[MAX_REG_OPERANDS];
        int src[MAX_REG_OPERANDS];
    } arch_reg;
    //int arch_reg[MAX_REG_OPERANDS]; // register number for bank conflict evaluation
    unsigned latency; // operation latency 
    unsigned initiation_interval;

    unsigned data_size; // what is the size of the word being operated on?
    memory_space_t space;
    cache_operator_type cache_op;

protected:
    bool m_decoded;
    virtual void pre_decode() {}
};

enum divergence_support_t {
   POST_DOMINATOR = 1,
   NUM_SIMD_MODEL
};

const unsigned MAX_ACCESSES_PER_INSN_PER_THREAD = 8;

class warp_inst_t: public inst_t {
public:
    // constructors
    warp_inst_t(int kain) 
    {
        m_uid=0;
        m_empty=true; 
        m_config=NULL; 
	//	printf("ressrve m_per_scalar_thread\n");
		m_per_scalar_thread.resize(32);
	//	printf("size of m)pe is %d\n",m_per_scalar_thread.size());
    }
    warp_inst_t() 
    {
        m_uid=0;
        m_empty=true; 
        m_config=NULL; 
		m_per_scalar_thread.resize(32);//KAN add here
    }
    warp_inst_t( const core_config *config ) 
    { 
        m_uid=0;
        assert(config->warp_size<=MAX_WARP_SIZE); 
        m_config=config;
        m_empty=true; 
        m_isatomic=false;
        m_per_scalar_thread_valid=false;
        m_mem_accesses_created=false;
        m_cache_hit=false;
		m_per_scalar_thread.resize(32);//KAN add here
    }
    virtual ~warp_inst_t(){
    }



	void warp_inst_copy_t(warp_inst_t &ori)
	{
		/*
		m_uid = ori.m_uid;
		m_empty = ori.m_empty;
		m_cache_hit = ori.m_cache_hit;
		issue_cycle = ori.issue_cycle;
		cycles  = ori.cycles; // used for implementing initiation interval delay
		m_isatomic = ori.m_isatomic;
		m_warp_id = ori.m_warp_id;
		m_config = ori.m_config;
		m_warp_active_mask = ori.m_warp_active_mask;
		m_per_scalar_thread_valid = ori.m_per_scalar_thread_valid;


		m_per_scalar_thread = ori.m_per_scalar_thread;
		//m_per_scalar_thread.assign(ori.m_per_scalar_thread.begin(),ori.m_per_scalar_thread.end());

		m_mem_accesses_created = ori.m_mem_accesses_created;
		m_accessq = ori.m_accessq;	
		sm_next_uid = ori.sm_next_uid;
		
//		//////////////////

    pc = ori.pc;        // program counter address of instruction
    isize = ori.isize;         // size of instruction in bytes 
    op = ori.op;             // opcode (uarch visible)
    memory_op = ori.memory_op; // memory_op used by ptxplus 

    reconvergence_pc = ori.reconvergence_pc; // -1 => not a branch, -2 => use function return address
   */ 
	for(int i = 0; i < 4; i++)
	{
    	out[i] = ori.out[i];
    	in[i] = ori.in[i];
	}
	
    is_vectorin = ori.is_vectorin;
    is_vectorout = ori.is_vectorout;
    pred = ori.pred; // predicate register number
    ar1 = ori.ar1;
	ar2 = ori.ar2; 
    // register number for bank conflict evaluation
	for(int i = 0; i < MAX_REG_OPERANDS; i++)
	{
		arch_reg.dst[i] = ori.arch_reg.dst[i];
		arch_reg.src[i] = ori.arch_reg.src[i];
	}
    latency = ori.latency; // operation latency 
    initiation_interval = ori.initiation_interval;

    data_size = ori.data_size; // what is the size of the word being operated on?
    space = ori.space;
    cache_op = ori.cache_op;
    m_decoded = ori.m_decoded;

	}





	void warp_inst_copy(warp_inst_t &ori)
	{
	//	m_uid = ori.m_uid;
	//	m_empty = ori.m_empty;
	//	m_cache_hit = ori.m_cache_hit;
	//	issue_cycle = ori.issue_cycle;
	//	cycles  = ori.cycles; // used for implementing initiation interval delay
		m_isatomic = ori.m_isatomic;
	//	m_warp_id = ori.m_warp_id;
		m_config = ori.m_config;
		m_warp_active_mask = ori.m_warp_active_mask;
		m_per_scalar_thread_valid = ori.m_per_scalar_thread_valid;


	//	m_per_scalar_thread = ori.m_per_scalar_thread;
		m_per_scalar_thread.assign(ori.m_per_scalar_thread.begin(),ori.m_per_scalar_thread.end());

		m_mem_accesses_created = ori.m_mem_accesses_created;
//		m_accessq = ori.m_accessq;	
		sm_next_uid = ori.sm_next_uid;
//		//////////////////

    pc = ori.pc;        // program counter address of instruction
    isize = ori.isize;         // size of instruction in bytes 
    op = ori.op;             // opcode (uarch visible)
    memory_op = ori.memory_op; // memory_op used by ptxplus 

    //reconvergence_pc = ori.reconvergence_pc; // -1 => not a branch, -2 => use function return address
    
	for(int i = 0; i < 4; i++)
	{
    	out[i] = ori.out[i];
    	in[i] = ori.in[i];
	}
    is_vectorin = ori.is_vectorin;
    is_vectorout = ori.is_vectorout;
    pred = ori.pred; // predicate register number
    ar1 = ori.ar1;
	ar2 = ori.ar2; 
    // register number for bank conflict evaluation
	for(int i = 0; i < MAX_REG_OPERANDS; i++)
	{
		arch_reg.dst[i] = ori.arch_reg.dst[i];
		arch_reg.src[i] = ori.arch_reg.src[i];
	}
    latency = ori.latency; // operation latency 
    initiation_interval = ori.initiation_interval;

    data_size = ori.data_size; // what is the size of the word being operated on?
    space = ori.space;
    cache_op = ori.cache_op;
    m_decoded = ori.m_decoded;

	}

    // modifiers
    void do_atomic(bool forceDo=false);
    void do_atomic( const active_mask_t& access_mask, bool forceDo=false );
    void clear() 
    { 
        m_empty=true; 
    }
    void issue( const active_mask_t &mask, unsigned warp_id, unsigned long long cycle ) 
    {
        m_warp_active_mask=mask;
        m_uid = ++sm_next_uid;
        m_warp_id = warp_id;
        issue_cycle = cycle;
        cycles = initiation_interval;
        m_cache_hit=false;
        m_empty=false;
    }
    void KAIN_copy_issue(  unsigned warp_id, unsigned long long cycle )
    {
        m_uid = ++sm_next_uid;
        m_warp_id = warp_id;
        issue_cycle = cycle;
        cycles = initiation_interval;
        m_cache_hit=false;
        m_empty=false;
    }



    void completed( unsigned long long cycle ) const;  // stat collection: called when the instruction is completed  
    void set_addr( unsigned n, new_addr_type addr ) 
    {
        if( !m_per_scalar_thread_valid ) {
          //  m_per_scalar_thread.resize(m_config->warp_size);
            m_per_scalar_thread_valid=true;
        }
        m_per_scalar_thread[n].memreqaddr[0] = addr;
    }
    void set_addr( unsigned n, new_addr_type* addr, unsigned num_addrs )
    {
        if( !m_per_scalar_thread_valid ) {
          //  m_per_scalar_thread.resize(m_config->warp_size);
            m_per_scalar_thread_valid=true;
        }
        assert(num_addrs <= MAX_ACCESSES_PER_INSN_PER_THREAD);
        for(unsigned i=0; i<num_addrs; i++)
            m_per_scalar_thread[n].memreqaddr[i] = addr[i];
    }

    struct transaction_info {
        std::bitset<4> chunks; // bitmask: 32-byte chunks accessed
        mem_access_byte_mask_t bytes;
        active_mask_t active; // threads in this transaction

        bool test_bytes(unsigned start_bit, unsigned end_bit) {
           for( unsigned i=start_bit; i<=end_bit; i++ )
              if(bytes.test(i))
                 return true;
           return false;
        }
    };

    void generate_mem_accesses();
    void memory_coalescing_arch_13( bool is_write, mem_access_type access_type );
    void memory_coalescing_arch_13_atomic( bool is_write, mem_access_type access_type );
    void memory_coalescing_arch_13_reduce_and_send( bool is_write, mem_access_type access_type, const transaction_info &info, new_addr_type addr, unsigned segment_size );

    void add_callback( unsigned lane_id, 
                       void (*function)(const class inst_t*, class ptx_thread_info*),
                       const inst_t *inst, 
                       class ptx_thread_info *thread )
    {
        if( !m_per_scalar_thread_valid ) {
          //  m_per_scalar_thread.resize(m_config->warp_size);
            m_per_scalar_thread_valid=true;
            m_isatomic=true;
        }
        m_per_scalar_thread[lane_id].callback.function = function;
        m_per_scalar_thread[lane_id].callback.instruction = inst;
        m_per_scalar_thread[lane_id].callback.thread = thread;
    }
    void set_active( const active_mask_t &active );

    void clear_active( const active_mask_t &inactive );
    void set_not_active( unsigned lane_id );

    // accessors
    virtual void print_insn(FILE *fp) const 
    {
        fprintf(fp," [inst @ pc=0x%04x] ", pc );
        for (int i=(int)m_config->warp_size-1; i>=0; i--)
            fprintf(fp, "%c", ((m_warp_active_mask[i])?'1':'0') );
    }
    bool active( unsigned thread ) const { return m_warp_active_mask.test(thread); }
    unsigned active_count() const { return m_warp_active_mask.count(); }
    bool empty() const { return m_empty; }
    unsigned warp_id() const 
    { 
        assert( !m_empty );
        return m_warp_id; 
    }
    bool has_callback( unsigned n ) const
    {
        return m_warp_active_mask[n] && m_per_scalar_thread_valid && 
            (m_per_scalar_thread[n].callback.function!=NULL);
    }
    new_addr_type get_addr( unsigned n ) const
    {
        assert( m_per_scalar_thread_valid );
        return m_per_scalar_thread[n].memreqaddr[0];
    }

    bool isatomic() const { return m_isatomic; }

    unsigned warp_size() const { return m_config->warp_size; }

    bool accessq_empty() const { return m_accessq.empty(); }
    unsigned accessq_count() const { return m_accessq.size(); }
    const mem_access_t &accessq_back() { return m_accessq.back(); }
    void accessq_pop_back() { m_accessq.pop_back(); }

    bool dispatch_delay()
    { 
        if( cycles > 0 ) 
            cycles--;
        return cycles > 0;
    }

    void print( FILE *fout ) const;
    unsigned get_uid() const { return m_uid; }

protected:

    unsigned m_uid;
    bool m_empty;
    bool m_cache_hit;
    unsigned long long issue_cycle;
    unsigned cycles; // used for implementing initiation interval delay
    bool m_isatomic;
    unsigned m_warp_id;
    const core_config *m_config; 
    active_mask_t m_warp_active_mask;

    struct per_thread_info {
        per_thread_info() {
            for(unsigned i=0; i<MAX_ACCESSES_PER_INSN_PER_THREAD; i++)
                memreqaddr[i] = 0;
        }
        dram_callback_t callback;
        new_addr_type memreqaddr[MAX_ACCESSES_PER_INSN_PER_THREAD]; // effective address, upto 8 different requests (to support 32B access in 8 chunks of 4B each)
    };
    bool m_per_scalar_thread_valid;
    std::vector<per_thread_info> m_per_scalar_thread;
    bool m_mem_accesses_created;
    std::list<mem_access_t> m_accessq;

    static unsigned sm_next_uid;
};

void move_warp( warp_inst_t *&dst, warp_inst_t *&src );

size_t get_kernel_code_size( class function_info *entry );

/*
 * This abstract class used as a base for functional and performance and simulation, it has basic functional simulation
 * data structures and procedures. 
 */
class core_t {
    public:
        virtual ~core_t() {}
        virtual void warp_exit( unsigned warp_id ) = 0;
        virtual bool warp_waiting_at_barrier( unsigned warp_id ) const = 0;
        virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t, unsigned tid)=0;
        class gpgpu_sim * get_gpu() {return m_gpu;}
        void execute_warp_inst_t(warp_inst_t &inst, unsigned warpSize, unsigned warpId =(unsigned)-1);
        bool  ptx_thread_done( unsigned hw_thread_id ) const ;
        void updateSIMTStack(unsigned warpId, unsigned warpSize, warp_inst_t * inst);
        void initilizeSIMTStack(unsigned warps, unsigned warpsSize);
        warp_inst_t getExecuteWarp(unsigned warpId);

    protected:
        class gpgpu_sim *m_gpu;
        kernel_info_t *m_kernel;
        simt_stack  **m_simt_stack; // pdom based reconvergence context for each warp
        class ptx_thread_info ** m_thread; 
};


//register that can hold multiple instructions.
class register_set {
public:
	register_set(unsigned num, const char* name){
		for( unsigned i = 0; i < num; i++ ) {
			regs.push_back(new warp_inst_t());
		}
		m_name = name;
	}
	bool has_free(){
		for( unsigned i = 0; i < regs.size(); i++ ) {
			if( regs[i]->empty() ) {
				return true;
			}
		}
		return false;
	}
	bool has_ready(){
		for( unsigned i = 0; i < regs.size(); i++ ) {
			if( not regs[i]->empty() ) {
				return true;
			}
		}
		return false;
	}

	void move_in( warp_inst_t *&src ){
		warp_inst_t** free = get_free();
		move_warp(*free, src);
	}
	//void copy_in( warp_inst_t* src ){
		//   src->copy_contents_to(*get_free());
		//}
	void move_out_to( warp_inst_t *&dest ){
		warp_inst_t **ready=get_ready();
		move_warp(dest, *ready);
	}

	warp_inst_t** get_ready(){
		warp_inst_t** ready;
		ready = NULL;
		for( unsigned i = 0; i < regs.size(); i++ ) {
			if( not regs[i]->empty() ) {
				if( ready and (*ready)->get_uid() < regs[i]->get_uid() ) {
					// ready is oldest
				} else {
					ready = &regs[i];
				}
			}
		}
		return ready;
	}

	void print(FILE* fp) const{
		fprintf(fp, "%s : @%p\n", m_name, this);
		for( unsigned i = 0; i < regs.size(); i++ ) {
			fprintf(fp, "     ");
			regs[i]->print(fp);
			fprintf(fp, "\n");
		}
	}

	warp_inst_t ** get_free(){
		for( unsigned i = 0; i < regs.size(); i++ ) {
			if( regs[i]->empty() ) {
				return &regs[i];
			}
		}
		assert(0 && "No free registers found");
		return NULL;
	}

private:
	std::vector<warp_inst_t*> regs;
	const char* m_name;
};

#endif // #ifdef __cplusplus

#endif // #ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED
