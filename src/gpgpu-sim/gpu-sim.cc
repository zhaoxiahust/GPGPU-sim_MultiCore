// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, George L. Yuan,
// Ali Bakhoda, Andrew Turner, Ivan Sham
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
#include <semaphore.h>
#include "gpu-sim.h"
#include <sys/time.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "zlib.h"

#include "../option_parser.h"
#include "shader.h"
#include "dram.h"
#include "mem_fetch.h"

#include <time.h>
#include "gpu-cache.h"
#include "gpu-misc.h"
#include "delayqueue.h"
#include "shader.h"
#include "icnt_wrapper.h"
#include "dram.h"
#include "addrdec.h"
#include "stat-tool.h"
#include "l2cache.h"

#include "../cuda-sim/ptx-stats.h"
#include "../intersim/statwraper.h"
#include "../intersim/interconnect_interface.h"
#include "../abstract_hardware_model.h"
#include "../debug.h"
#include "../gpgpusim_entrypoint.h"
#include "../cuda-sim/cuda-sim.h"

#include "mem_latency_stat.h"
#include "visualizer.h"
#include "stats.h"

#include <stdio.h>
#include <string.h>

#define MAX(a,b) (((a)>(b))?(a):(b))

bool g_interactive_debugger_enabled=false;

unsigned long long  gpu_sim_cycle = 0;
unsigned long long  gpu_tot_sim_cycle = 0;

// performance counter for stalls due to congestion.
unsigned int gpu_stall_dramfull = 0; 
unsigned int gpu_stall_icnt2sh = 0;

/* Clock Domains */

#define  CORE  0x01
#define  L2    0x02
#define  DRAM  0x04
#define  ICNT  0x08  


#define MEM_LATENCY_STAT_IMPL
#include "mem_latency_stat.h"

void memory_config::reg_options(class OptionParser * opp)
{
    option_parser_register(opp, "-gpgpu_dram_scheduler", OPT_INT32, &scheduler_type, 
                                "0 = fifo, 1 = FR-FCFS (defaul)", "1");
    option_parser_register(opp, "-gpgpu_dram_partition_queues", OPT_CSTR, &gpgpu_L2_queue_config, 
                           "i2$:$2d:d2$:$2i",
                           "8:8:8:8");

    option_parser_register(opp, "-l2_ideal", OPT_BOOL, &l2_ideal, 
                           "Use a ideal L2 cache that always hit",
                           "0");
    option_parser_register(opp, "-gpgpu_cache:dl2", OPT_CSTR, &m_L2_config.m_config_string, 
                   "unified banked L2 data cache config "
                   " {<nsets>:<bsize>:<assoc>:<rep>:<wr>:<alloc>,<mshr>:<N>:<merge>,<mq>}",
                   "64:128:8:L:R:m,A:16:4,4");
    option_parser_register(opp, "-gpgpu_cache:dl2_texture_only", OPT_BOOL, &m_L2_texure_only, 
                           "L2 cache used for texture only",
                           "1");
    option_parser_register(opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem, 
                 "number of memory modules (e.g. memory controllers) in gpu",
                 "8");
    option_parser_register(opp, "-gpgpu_n_mem_per_ctrlr", OPT_UINT32, &gpu_n_mem_per_ctrlr, 
                 "number of memory chips per memory controller",
                 "1");
    option_parser_register(opp, "-gpgpu_memlatency_stat", OPT_INT32, &gpgpu_memlatency_stat, 
                "track and display latency statistics 0x2 enables MC, 0x4 enables queue logs",
                "0");
    option_parser_register(opp, "-gpgpu_dram_sched_queue_size", OPT_INT32, &gpgpu_dram_sched_queue_size, 
                "0 = unlimited (default); # entries per chip",
                "0");
    option_parser_register(opp, "-gpgpu_dram_buswidth", OPT_UINT32, &busW, 
                 "default = 4 bytes (8 bytes per cycle at DDR)",
                 "4");
    option_parser_register(opp, "-gpgpu_dram_burst_length", OPT_UINT32, &BL, 
                 "Burst length of each DRAM request (default = 4 DDR cycle)",
                 "4");
    option_parser_register(opp, "-gpgpu_dram_timing_opt", OPT_CSTR, &gpgpu_dram_timing_opt, 
                "DRAM timing parameters = {nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tCDLR:tWR:nbkgrp:tCCDL:tRTPL}",
                "4:2:8:12:21:13:34:9:4:5:13:1:0:0");
    option_parser_register(opp, "-rop_latency", OPT_UINT32, &rop_latency,
                     "ROP queue latency (default 85)",
                     "85");
    option_parser_register(opp, "-dram_latency", OPT_UINT32, &dram_latency,
                     "DRAM latency (default 30)",
                     "30");

    m_address_mapping.addrdec_setoption(opp);
}

void shader_core_config::reg_options(class OptionParser * opp)
{
    option_parser_register(opp, "-gpgpu_simd_model", OPT_INT32, &model, 
                   "1 = post-dominator", "1");
    option_parser_register(opp, "-gpgpu_shader_core_pipeline", OPT_CSTR, &gpgpu_shader_core_pipeline_opt, 
                   "shader core pipeline config, i.e., {<nthread>:<warpsize>}",
                   "1024:32");
    option_parser_register(opp, "-gpgpu_tex_cache:l1", OPT_CSTR, &m_L1T_config.m_config_string, 
                   "per-shader L1 texture cache  (READ-ONLY) config "
                   " {<nsets>:<bsize>:<assoc>:<rep>:<wr>:<alloc>,<mshr>:<N>:<merge>,<mq>:<rf>}",
                   "8:128:5:L:R:m,F:128:4,128:2");
    option_parser_register(opp, "-gpgpu_const_cache:l1", OPT_CSTR, &m_L1C_config.m_config_string, 
                   "per-shader L1 constant memory cache  (READ-ONLY) config "
                   " {<nsets>:<bsize>:<assoc>:<rep>:<wr>:<alloc>,<mshr>:<N>:<merge>,<mq>}",
                   "64:64:2:L:R:f,A:2:32,4" );
    option_parser_register(opp, "-gpgpu_cache:il1", OPT_CSTR, &m_L1I_config.m_config_string, 
                   "shader L1 instruction cache config "
                   " {<nsets>:<bsize>:<assoc>:<rep>:<wr>:<alloc>,<mshr>:<N>:<merge>,<mq>}",
                   "4:256:4:L:R:f,A:2:32,4" );
    option_parser_register(opp, "-gpgpu_cache:dl1", OPT_CSTR, &m_L1D_config.m_config_string, 
                   "per-shader L1 data cache config "
                   " {<nsets>:<bsize>:<assoc>:<rep>:<wr>:<alloc>,<mshr>:<N>:<merge>,<mq>|none}",
                   "none" );
    option_parser_register(opp, "-gpgpu_perfect_mem", OPT_BOOL, &gpgpu_perfect_mem, 
                 "enable perfect memory mode (no cache miss)",
                 "0");
    option_parser_register(opp, "-gpgpu_shader_registers", OPT_UINT32, &gpgpu_shader_registers, 
                 "Number of registers per shader core. Limits number of concurrent CTAs. (default 8192)",
                 "8192");
    option_parser_register(opp, "-gpgpu_shader_cta", OPT_UINT32, &max_cta_per_core, 
                 "Maximum number of concurrent CTAs in shader (default 8)",
                 "8");
    option_parser_register(opp, "-gpgpu_n_clusters", OPT_UINT32, &n_simt_clusters, 
                 "number of processing clusters",
                 "10");
    option_parser_register(opp, "-gpgpu_n_cores_per_cluster", OPT_UINT32, &n_simt_cores_per_cluster, 
                 "number of simd cores per cluster",
                 "3");
    option_parser_register(opp, "-gpgpu_n_cluster_ejection_buffer_size", OPT_UINT32, &n_simt_ejection_buffer_size, 
                 "number of packets in ejection buffer",
                 "8");
    option_parser_register(opp, "-gpgpu_n_ldst_response_buffer_size", OPT_UINT32, &ldst_unit_response_queue_size, 
                 "number of response packets in ld/st unit ejection buffer",
                 "2");
    option_parser_register(opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_size, 
                 "Size of shared memory per shader core (default 16kB)",
                 "16384");
    option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32, &mem_warp_parts,  
                 "Number of portions a warp is divided into for shared memory bank conflict check ",
                 "2");
    option_parser_register(opp, "-gpgpu_warpdistro_shader", OPT_INT32, &gpgpu_warpdistro_shader, 
                "Specify which shader core to collect the warp size distribution from", 
                "-1");
    option_parser_register(opp, "-gpgpu_local_mem_map", OPT_BOOL, &gpgpu_local_mem_map, 
                "Mapping from local memory space address to simulated GPU physical address space (default = enabled)", 
                "1");
    option_parser_register(opp, "-gpgpu_num_reg_banks", OPT_INT32, &gpgpu_num_reg_banks, 
                "Number of register banks (default = 8)", 
                "8");
    option_parser_register(opp, "-gpgpu_reg_bank_use_warp_id", OPT_BOOL, &gpgpu_reg_bank_use_warp_id,
             "Use warp ID in mapping registers to banks (default = off)",
             "0");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_sp", OPT_INT32, &gpgpu_operand_collector_num_units_sp,
                "number of collector units (default = 4)", 
                "4");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_sfu", OPT_INT32, &gpgpu_operand_collector_num_units_sfu,
                "number of collector units (default = 4)", 
                "4");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_mem", OPT_INT32, &gpgpu_operand_collector_num_units_mem,
                "number of collector units (default = 2)", 
                "2");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_gen", OPT_INT32, &gpgpu_operand_collector_num_units_gen,
                "number of collector units (default = 0)", 
                "0");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sp", OPT_INT32, &gpgpu_operand_collector_num_in_ports_sp,
                           "number of collector unit in ports (default = 1)", 
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sfu", OPT_INT32, &gpgpu_operand_collector_num_in_ports_sfu,
                           "number of collector unit in ports (default = 1)", 
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_mem", OPT_INT32, &gpgpu_operand_collector_num_in_ports_mem,
                           "number of collector unit in ports (default = 1)", 
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_gen", OPT_INT32, &gpgpu_operand_collector_num_in_ports_gen,
                           "number of collector unit in ports (default = 0)", 
                           "0");
    option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sp", OPT_INT32, &gpgpu_operand_collector_num_out_ports_sp,
                           "number of collector unit in ports (default = 1)", 
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sfu", OPT_INT32, &gpgpu_operand_collector_num_out_ports_sfu,
                           "number of collector unit in ports (default = 1)", 
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_mem", OPT_INT32, &gpgpu_operand_collector_num_out_ports_mem,
                           "number of collector unit in ports (default = 1)", 
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_gen", OPT_INT32, &gpgpu_operand_collector_num_out_ports_gen,
                           "number of collector unit in ports (default = 0)", 
                           "0");
    option_parser_register(opp, "-gpgpu_coalesce_arch", OPT_INT32, &gpgpu_coalesce_arch, 
                            "Coalescing arch (default = 13, anything else is off for now)", 
                            "13");
    option_parser_register(opp, "-gpgpu_num_sched_per_core", OPT_INT32, &gpgpu_num_sched_per_core, 
                            "Number of warp schedulers per core", 
                            "1");
    option_parser_register(opp, "-gpgpu_max_insn_issue_per_warp", OPT_INT32, &gpgpu_max_insn_issue_per_warp,
                            "Max number of instructions that can be issued per warp in one cycle by scheduler",
                            "2");
    option_parser_register(opp, "-gpgpu_simt_core_sim_order", OPT_INT32, &simt_core_sim_order,
                            "Select the simulation order of cores in a cluster (0=Fix, 1=Round-Robin)",
                            "1");
    option_parser_register(opp, "-gpgpu_pipeline_widths", OPT_CSTR, &pipeline_widths_string,
                            "Pipeline widths "
    		                "ID_OC_SP,ID_OC_SFU,ID_OC_MEM,OC_EX_SP,OC_EX_SFU,OC_EX_MEM,EX_WB",
                            "1,1,1,1,1,1,1" );
    option_parser_register(opp, "-gpgpu_num_sp_units", OPT_INT32, &gpgpu_num_sp_units,
    		                "Number of SP units (default=1)",
    		                "1");
    option_parser_register(opp, "-gpgpu_num_sfu_units", OPT_INT32, &gpgpu_num_sfu_units,
    		                "Number of SF units (default=1)",
    		                "1");
    option_parser_register(opp, "-gpgpu_num_mem_units", OPT_INT32, &gpgpu_num_mem_units,
    		                "Number if ldst units (default=1) WARNING: not hooked up to anything",
                             "1");
}

void gpgpu_sim_config::reg_options(option_parser_t opp)
{
    gpgpu_functional_sim_config::reg_options(opp);
    m_shader_config.reg_options(opp);
    m_memory_config.reg_options(opp);

   option_parser_register(opp, "-gpgpu_max_cycle", OPT_INT32, &gpu_max_cycle_opt, 
               "terminates gpu simulation early (0 = no limit)",
               "0");
   option_parser_register(opp, "-gpgpu_max_insn", OPT_INT32, &gpu_max_insn_opt, 
               "terminates gpu simulation early (0 = no limit)",
               "0");
   option_parser_register(opp, "-gpgpu_max_cta", OPT_INT32, &gpu_max_cta_opt, 
               "terminates gpu simulation early (0 = no limit)",
               "0");
   option_parser_register(opp, "-gpgpu_runtime_stat", OPT_CSTR, &gpgpu_runtime_stat, 
                  "display runtime statistics such as dram utilization {<freq>:<flag>}",
                  "10000:0");
   option_parser_register(opp, "-gpgpu_flush_cache", OPT_BOOL, &gpgpu_flush_cache, 
                "Flush cache at the end of each kernel call",
                "0");
   option_parser_register(opp, "-gpgpu_deadlock_detect", OPT_BOOL, &gpu_deadlock_detect, 
                "Stop the simulation at deadlock (1=on (default), 0=off)", 
                "1");
   option_parser_register(opp, "-gpgpu_ptx_instruction_classification", OPT_INT32, 
               &gpgpu_ptx_instruction_classification, 
               "if enabled will classify ptx instruction types per kernel (Max 255 kernels now)", 
               "0");
   option_parser_register(opp, "-gpgpu_ptx_sim_mode", OPT_INT32, &g_ptx_sim_mode, 
               "Select between Performance (default) or Functional simulation (1)", 
               "0");
   option_parser_register(opp, "-gpgpu_clock_domains", OPT_CSTR, &gpgpu_clock_domains, 
                  "Clock Domain Frequencies in MhZ {<Core Clock>:<ICNT Clock>:<L2 Clock>:<DRAM Clock>}",
                  "500.0:2000.0:2000.0:2000.0");
   option_parser_register(opp, "-gpgpu_max_concurrent_kernel", OPT_INT32, &max_concurrent_kernel,
                          "maximum kernels that can run concurrently on GPU", "8" );
   option_parser_register(opp, "-gpgpu_cflog_interval", OPT_INT32, &gpgpu_cflog_interval, 
               "Interval between each snapshot in control flow logger", 
               "0");
   option_parser_register(opp, "-visualizer_enabled", OPT_BOOL,
                          &g_visualizer_enabled, "Turn on visualizer output (1=On, 0=Off)",
                          "1");
   option_parser_register(opp, "-visualizer_outputfile", OPT_CSTR, 
                          &g_visualizer_filename, "Specifies the output log file for visualizer",
                          NULL);
   option_parser_register(opp, "-visualizer_zlevel", OPT_INT32,
                          &g_visualizer_zlevel, "Compression level of the visualizer output log (0=no comp, 9=highest)",
                          "6");
   ptx_file_line_stats_options(opp);
}

/////////////////////////////////////////////////////////////////////////////

void increment_x_then_y_then_z( dim3 &i, const dim3 &bound)
{
   i.x++;
   if ( i.x >= bound.x ) {
      i.x = 0;
      i.y++;
      if ( i.y >= bound.y ) {
         i.y = 0;
         if( i.z < bound.z ) 
            i.z++;
      }
   }
}

void gpgpu_sim::launch( kernel_info_t *kinfo )
{
   //KAIN add
   for(int n=0; n < m_running_kernels.size(); n++ ) {
       if( (NULL!=m_running_kernels[n]) &&(~ m_running_kernels[n]->done()) ) {
	   printf("alraedy a kernel is runnning\n");
	   assert(0);
       }
   }
   //KAIN add
   unsigned cta_size = kinfo->threads_per_cta();
   if ( cta_size > m_shader_config->n_thread_per_shader ) {
      printf("Execution error: Shader kernel CTA (block) size is too large for microarch config.\n");
      printf("                 CTA size (x*y*z) = %u, max supported = %u\n", cta_size, 
             m_shader_config->n_thread_per_shader );
      printf("                 => either change -gpgpu_shader argument in gpgpusim.config file or\n");
      printf("                 modify the CUDA source to decrease the kernel block size.\n");
      abort();
   }
   unsigned n=0;
   for(n=0; n < m_running_kernels.size(); n++ ) {
       if( (NULL==m_running_kernels[n]) || m_running_kernels[n]->done() ) {
           m_running_kernels[n] = kinfo;
           break;
       }
   }
   assert(n < m_running_kernels.size());
}

bool gpgpu_sim::can_start_kernel()
{
   for(unsigned n=0; n < m_running_kernels.size(); n++ ) {
       if( (NULL==m_running_kernels[n]) || m_running_kernels[n]->done() ) 
           return true;
   }
   return false;
}
extern int End_Block_Process[5];
extern int Process_id;
extern int Process_count;
bool gpgpu_sim::get_more_cta_left() const
{ 
   if (m_config.gpu_max_cta_opt != 0) {
      if( m_total_cta_launched >= m_config.gpu_max_cta_opt )
          return false;
   }
   for(unsigned n=0; n < m_running_kernels.size(); n++ ) {
       if( m_running_kernels[n] && !m_running_kernels[n]->no_more_ctas_to_run() ) 
           return true;
   }
   /*
   static int first = 0;
   if(first == 0 && End_Block_Process[Process_id] != 0)
   {
  		first = 1; 
		if(Process_id != Process_count-1)
			printf("no block to launch, cycle is %lld\n",gpu_tot_sim_cycle + gpu_sim_cycle);
   }
   */
   return false;
}

kernel_info_t *gpgpu_sim::select_kernel()
{
    for(unsigned n=0; n < m_running_kernels.size(); n++ ) {
        unsigned idx = (n+m_last_issued_kernel+1)%m_config.max_concurrent_kernel;
        if( m_running_kernels[idx] && !m_running_kernels[idx]->no_more_ctas_to_run() ) {
            m_last_issued_kernel=idx;
            return m_running_kernels[idx];
        }
    }
    return NULL;
}
extern int Process_id;
unsigned gpgpu_sim::finished_kernel()
{
    if( m_finished_kernel.empty() ) 
        return 0;
    unsigned result = m_finished_kernel.front();
    m_finished_kernel.pop_front();
	printf("Process %d, popfindished kernel\n",Process_id);
    return result;
}

void gpgpu_sim::set_kernel_done( kernel_info_t *kernel ) 
{ 
	

	printf("Process %d, push findished kernel\n",Process_id);
    unsigned uid = kernel->get_uid();
    m_finished_kernel.push_back(uid);
    std::vector<kernel_info_t*>::iterator k;
    for( k=m_running_kernels.begin(); k!=m_running_kernels.end(); k++ ) {
        if( *k == kernel ) {
            *k = NULL;
            break;
        }
    }
    assert( k != m_running_kernels.end() ); 
}

void set_ptx_warp_size(const struct core_config * warp_size);

gpgpu_sim::gpgpu_sim( const gpgpu_sim_config &config ) 
    : gpgpu_t(config), m_config(config)
{ 
    m_shader_config = &m_config.m_shader_config;
    m_memory_config = &m_config.m_memory_config;
    set_ptx_warp_size(m_shader_config);
    ptx_file_line_stats_create_exposed_latency_tracker(m_config.num_shader());

    m_shader_stats = new shader_core_stats(m_shader_config);
    m_memory_stats = new memory_stats_t(m_config.num_shader(),m_shader_config,m_memory_config);

    gpu_sim_insn = 0;
    gpu_tot_sim_insn = 0;
    gpu_tot_issued_cta = 0;
    gpu_deadlock = false;

    m_cluster = new simt_core_cluster*[m_shader_config->n_simt_clusters];
    for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) 
    {
        m_cluster[i] = new simt_core_cluster(this,i,m_shader_config,m_memory_config,m_shader_stats,m_memory_stats);
        //m_cluster[i]->KAINsetThreadID(0);// KAIN set here to support the sigle thread performance
        m_cluster[i]->m_KAIN_process.clear();
    }

    m_memory_partition_unit = new memory_partition_unit*[m_memory_config->m_n_mem];
    for (unsigned i=0;i<m_memory_config->m_n_mem;i++) 
        m_memory_partition_unit[i] = new memory_partition_unit(i, m_memory_config, m_memory_stats);

    icnt_init(m_shader_config->n_simt_clusters,m_memory_config->m_n_mem);

    time_vector_create(NUM_MEM_REQ_STAT);
    fprintf(stdout, "GPGPU-Sim uArch: performance model initialization complete.\n");

    m_running_kernels.resize( config.max_concurrent_kernel, NULL );
    m_last_issued_kernel = 0;
    m_last_cluster_issue = 0;
}

int gpgpu_sim::shared_mem_size() const
{
   return m_shader_config->gpgpu_shmem_size;
}

int gpgpu_sim::num_registers_per_core() const
{
   return m_shader_config->gpgpu_shader_registers;
}

int gpgpu_sim::wrp_size() const
{
   return m_shader_config->warp_size;
}

int gpgpu_sim::shader_clock() const
{
   return m_config.core_freq/1000;
}

void gpgpu_sim::set_prop( cudaDeviceProp *prop )
{
   m_cuda_properties = prop;
}

const struct cudaDeviceProp *gpgpu_sim::get_prop() const
{
   return m_cuda_properties;
}

enum divergence_support_t gpgpu_sim::simd_model() const
{
   return m_shader_config->model;
}

void gpgpu_sim_config::init_clock_domains(void ) 
{
   sscanf(gpgpu_clock_domains,"%lf:%lf:%lf:%lf", 
          &core_freq, &icnt_freq, &l2_freq, &dram_freq);
   core_freq = core_freq MhZ;
   icnt_freq = icnt_freq MhZ;
   l2_freq = l2_freq MhZ;
   dram_freq = dram_freq MhZ;        
   core_period = 1/core_freq;
   icnt_period = 1/icnt_freq;
   dram_period = 1/dram_freq;
   l2_period = 1/l2_freq;
   printf("GPGPU-Sim uArch: clock freqs: %lf:%lf:%lf:%lf\n",core_freq,icnt_freq,l2_freq,dram_freq);
   printf("GPGPU-Sim uArch: clock periods: %.20lf:%.20lf:%.20lf:%.20lf\n",core_period,icnt_period,l2_period,dram_period);
}

void gpgpu_sim::reinit_clock_domains(void)
{
   core_time = 0;
   dram_time = 0;
   icnt_time = 0;
   l2_time = 0;
}

bool gpgpu_sim::active()
{
    if (m_config.gpu_max_cycle_opt && (gpu_tot_sim_cycle + gpu_sim_cycle) >= m_config.gpu_max_cycle_opt) 
       return false;
    if (m_config.gpu_max_insn_opt && (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt) 
       return false;
    if (m_config.gpu_max_cta_opt && (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt) )
       return false;
    if (m_config.gpu_deadlock_detect && gpu_deadlock) 
       return false;
    for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) 
       if( m_cluster[i]->get_not_completed()>0 ) 
           return true;;
    for (unsigned i=0;i<m_memory_config->m_n_mem;i++) 
       if( m_memory_partition_unit[i]->busy()>0 )
           return true;;
    if( icnt_busy() )
        return true;
    if( get_more_cta_left() )
        return true;
    return false;
}

void gpgpu_sim::init()
{
    // run a CUDA grid on the GPU microarchitecture simulator
    gpu_sim_cycle = 0;
    gpu_sim_insn = 0;
    last_gpu_sim_insn = 0;
    m_total_cta_launched=0;

    reinit_clock_domains();
    set_param_gpgpu_num_shaders(m_config.num_shader());
    for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) 
       m_cluster[i]->reinit();
    m_shader_stats->new_grid();
    // initialize the control-flow, memory access, memory latency logger
    if (m_config.g_visualizer_enabled) {
        create_thread_CFlogger( m_config.num_shader(), m_shader_config->n_thread_per_shader, 0, m_config.gpgpu_cflog_interval );
    }
    shader_CTA_count_create( m_config.num_shader(), m_config.gpgpu_cflog_interval);
    if (m_config.gpgpu_cflog_interval != 0) {
       insn_warp_occ_create( m_config.num_shader(), m_shader_config->warp_size );
       shader_warp_occ_create( m_config.num_shader(), m_shader_config->warp_size, m_config.gpgpu_cflog_interval);
       shader_mem_acc_create( m_config.num_shader(), m_memory_config->m_n_mem, 4, m_config.gpgpu_cflog_interval);
       shader_mem_lat_create( m_config.num_shader(), m_config.gpgpu_cflog_interval);
       shader_cache_access_create( m_config.num_shader(), 3, m_config.gpgpu_cflog_interval);
       set_spill_interval (m_config.gpgpu_cflog_interval * 40);
    }

    if (g_network_mode) 
       icnt_init_grid(); 
}

void gpgpu_sim::update_stats() {
    m_memory_stats->memlatstat_lat_pw();
    gpu_tot_sim_cycle += gpu_sim_cycle;
    gpu_tot_sim_insn += gpu_sim_insn;
}

void gpgpu_sim::print_stats()
{

    ptx_file_line_stats_write_file();
    gpu_print_stat();
/*KAIN delete this
    if (g_network_mode) {
       interconnect_stats();
       printf("----------------------------Interconnect-DETAILS---------------------------------" );
       icnt_overal_stat();
       printf("----------------------------END-of-Interconnect-DETAILS-------------------------" );
    }
*/
	
}

void gpgpu_sim::deadlock_check()
{
   if (m_config.gpu_deadlock_detect && gpu_deadlock) {
      fflush(stdout);
      printf("\n\nGPGPU-Sim uArch: ERROR ** deadlock detected: last writeback core %u @ gpu_sim_cycle %u (+ gpu_tot_sim_cycle %u) (%u cycles ago)\n", 
             gpu_sim_insn_last_update_sid,
             (unsigned) gpu_sim_insn_last_update, (unsigned) (gpu_tot_sim_cycle-gpu_sim_cycle),
             (unsigned) (gpu_sim_cycle - gpu_sim_insn_last_update )); 
      unsigned num_cores=0;
      for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
         unsigned not_completed = m_cluster[i]->get_not_completed();
         if( not_completed ) {
             if ( !num_cores )  {
                 printf("GPGPU-Sim uArch: DEADLOCK  shader cores no longer committing instructions [core(# threads)]:\n" );
                 printf("GPGPU-Sim uArch: DEADLOCK  ");
                 m_cluster[i]->print_not_completed(stdout);
             } else if (num_cores < 8 ) {
                 m_cluster[i]->print_not_completed(stdout);
             } else if (num_cores >= 8 ) {
                 printf(" + others ... ");
             }
             num_cores+=m_shader_config->n_simt_cores_per_cluster;
         }
      }
      printf("\n");
      for (unsigned i=0;i<m_memory_config->m_n_mem;i++) {
         bool busy = m_memory_partition_unit[i]->busy();
         if( busy ) 
             printf("GPGPU-Sim uArch DEADLOCK:  memory partition %u busy\n", i );
      }
      if( icnt_busy() ) {
         printf("GPGPU-Sim uArch DEADLOCK:  iterconnect contains traffic\n");
         display_icnt_state( stdout );
      }
      printf("\nRe-run the simulator in gdb and use debug routines in .gdbinit to debug this\n");
      fflush(stdout);
      abort();
   }
}

void gpgpu_sim::gpu_print_stat() const
{  
    extern double time_core1;
    extern double time_icnt1;
    extern double time_dram;
    extern double time_l2;
    extern double time_icnt2;
    extern double time_core2;
	extern double time_last_thread;

    extern double time_core_issue;
    extern double wb_time[20];
    extern double ex_time[20];
    extern double read_time[20];
    extern double issue_time[20];
    extern double decode_time[20];
    extern double fetch_time[20];
	extern double time_cluster_thread0[20];

    printf("core1 time is %.20lf\n",time_core1);
    printf("icnt1 time is %.20lf\n",time_icnt1);
    printf("dram time is %.20lf\n",time_dram);
    printf("l2 time is %.20lf\n",time_l2);
    printf("icnt2 time is %.20lf\n",time_icnt2);
    printf("core2 time is %.20lf\n",time_core2);
    printf("first over thread time is %.20lf\n",time_last_thread);
    printf("core_issue time is %.20lf\n",time_core_issue);
	for(int i = 0; i < Cluster_Thread_Num; i++)
	{
    	printf("Cluster-thread %d, time is %.20lf\n",i,time_cluster_thread0[i]);
		printf("Cluster-thread0 wb time is %.20lf\n",wb_time[i]);
		printf("Cluster-thread0 ex time is %.20lf\n",ex_time[i]);
		printf("Cluster-thread0 read operand time is %.20lf\n",read_time[i]);
		printf("Cluster-thread0 issue time is %.20lf\n",issue_time[i]);
		printf("Cluster-thread0 decode time is %.20lf\n",decode_time[i]);
		printf("Cluster-thread0 fetch time is %.20lf\n",fetch_time[i]);
	}



   printf("gpu_sim_cycle = %lld\n", gpu_sim_cycle);
   printf("gpu_sim_insn = %lld\n", gpu_sim_insn);
   printf("gpu_ipc = %12.4f\n", (float)gpu_sim_insn / gpu_sim_cycle);
   printf("gpu_tot_sim_cycle = %lld\n", gpu_tot_sim_cycle+gpu_sim_cycle);
   printf("gpu_tot_sim_insn = %lld\n", gpu_tot_sim_insn+gpu_sim_insn);
   printf("gpu_tot_ipc = %12.4f\n", (float)(gpu_tot_sim_insn+gpu_sim_insn) / (gpu_tot_sim_cycle+gpu_sim_cycle));
   printf("gpu_tot_issued_cta = %lld\n", gpu_tot_issued_cta);

   // performance counter for stalls due to congestion.
   printf("gpu_stall_dramfull = %d\n", gpu_stall_dramfull);
   printf("gpu_stall_icnt2sh    = %d\n", gpu_stall_icnt2sh );

   time_t curr_time;
   time(&curr_time);
   unsigned long long elapsed_time = MAX( curr_time - g_simulation_starttime, 1 );
   printf( "gpu_total_sim_rate=%u\n", (unsigned)( ( gpu_tot_sim_insn + gpu_sim_insn ) / elapsed_time ) );

   shader_print_l1_miss_stat( stdout );

   m_shader_stats->print(stdout);

   // performance counter that are not local to one shader
   m_memory_stats->memlatstat_print(m_memory_config->m_n_mem,m_memory_config->nbk);
   m_memory_stats->print(stdout);
   for (unsigned i=0;i<m_memory_config->m_n_mem;i++) 
      m_memory_partition_unit[i]->print(stdout);
   if (!m_memory_config->m_L2_config.disabled() && m_memory_config->m_L2_config.get_num_lines())
      L2c_print_cache_stat();
   if (m_config.gpgpu_cflog_interval != 0) {
      spill_log_to_file (stdout, 1, gpu_sim_cycle);
      insn_warp_occ_print(stdout);
   }
   if ( gpgpu_ptx_instruction_classification ) {
      StatDisp( g_inst_classification_stat[g_ptx_kernel_count]);
      StatDisp( g_inst_op_classification_stat[g_ptx_kernel_count]);
   }
   time_vector_print();
   fflush(stdout);
}


// performance counter that are not local to one shader
unsigned gpgpu_sim::threads_per_core() const 
{ 
   return m_shader_config->n_thread_per_shader; 
}

void shader_core_ctx::mem_instruction_stats(const warp_inst_t &inst)
{
    unsigned active_count = inst.active_count(); 
    //this breaks some encapsulation: the is_[space] functions, if you change those, change this.
    switch (inst.space.get_type()) {
    case undefined_space:
    case reg_space:
        break;
    case shared_space:
        m_stats->gpgpu_n_shmem_insn += active_count; 
        break;
    case const_space:
        m_stats->gpgpu_n_const_insn += active_count;
        break;
    case param_space_kernel:
    case param_space_local:
        m_stats->gpgpu_n_param_insn += active_count;
        break;
    case tex_space:
        m_stats->gpgpu_n_tex_insn += active_count;
        break;
    case global_space:
    case local_space:
        if( inst.is_store() )
            m_stats->gpgpu_n_store_insn += active_count;
        else 
            m_stats->gpgpu_n_load_insn += active_count;
        break;
    default:
        abort();
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Launches a cooperative thread array (CTA). 
 *  
 * @param kernel 
 *    object that tells us which kernel to ask for a CTA from 
 */
//int CTAperShader_kain;
extern class KAIN_warp_inst **block_warpid[299999][100];
void shader_core_ctx::issue_block2core( kernel_info_t &kernel ) 
{
   // printf("in set max cta\n");
    set_max_cta(kernel);
//	assert(kernel_max_cta_per_shader*AllCores<kernel.num_blocks());//KAIN, to have the enough space to do the spool
  //  printf("out set max cta\n");

    // find a free CTA context 
    unsigned free_cta_hw_id=(unsigned)-1;
    for (unsigned i=0;i<kernel_max_cta_per_shader;i++ ) {
      if( m_cta_status[i]==0 ) {
         free_cta_hw_id=i;
         break;
      }
    }
    assert( free_cta_hw_id!=(unsigned)-1 );

    // determine hardware threads and warps that will be used for this CTA
    int cta_size = kernel.threads_per_cta();

    // hw warp id = hw thread id mod warp size, so we need to find a range 
    // of hardware thread ids corresponding to an integral number of hardware
    // thread ids
    int padded_cta_size = cta_size; 
    if (cta_size%m_config->warp_size)
      padded_cta_size = ((cta_size/m_config->warp_size)+1)*(m_config->warp_size);
    unsigned start_thread = free_cta_hw_id * padded_cta_size;
    unsigned end_thread  = start_thread +  cta_size;

    // reset the microarchitecture state of the selected hardware thread and warp contexts
    reinit(start_thread, end_thread,false);
     
    // initalize scalar threads and determine which hardware warps they are allocated to
    // bind functional simulation state of threads to hardware resources (simulation) 
    warp_set_t warps;
    unsigned nthreads_in_block= 0;
    int ID = KAINgetThreadID();//KAIN
    //////////////////////////////??KAIN free the memory,I think this memory can be used by other kernels
    /*
    if(m_thread[start_thread] != NULL){
        unsigned BlockID;
        unsigned WarpID;
        unsigned nWarps;
        if(cta_size % m_config->warp_size)
            nWarps = cta_size / m_config->warp_size + 1;
        else
            nWarps = cta_size / m_config->warp_size;
        m_thread[start_thread]->KAIN_get_cta_num(m_config->warp_size,0,&BlockID,&WarpID); 
        printf("Free CTA %d\n",BlockID);
        for(int i = 0; i < nWarps; i++)
            delete block_warpid[BlockID][i];
        /////////////////////
    }
    */
//	printf("before init in Cycle\n");
//	fflush(stdout);
//	CTAperShader_kain = kernel_max_cta_per_shader;
    for (unsigned i = start_thread; i<end_thread; i++) {
        m_threadState[i].m_cta_id = free_cta_hw_id;
        unsigned warp_id = i/m_config->warp_size;

     //   printf("in init\n");
        nthreads_in_block += ptx_sim_init_thread1(kernel,&m_thread[i],m_sid,i,cta_size-(i-start_thread),m_config->n_thread_per_shader,this,free_cta_hw_id,warp_id,m_cluster->get_gpu(),kernel_max_cta_per_shader);
         m_thread[i]->set_ThreadID_kain(ID);
     //   printf("out init \n");
        m_threadState[i].m_active = true; 
        warps.set( warp_id );
    }

//	printf("after init in Cycle\n");
//	fflush(stdout);
    assert( nthreads_in_block > 0 && nthreads_in_block <= m_config->n_thread_per_shader); // should be at least one, but less than max
    m_cta_status[free_cta_hw_id]=nthreads_in_block;

    // now that we know which warps are used in this CTA, we can allocate
    // resources for use in CTA-wide barrier operations
    m_barriers.allocate_barrier(free_cta_hw_id,warps);

    // initialize the SIMT stacks and fetch hardware
    init_warps( free_cta_hw_id, start_thread, end_thread);
    m_n_active_cta++;

    shader_CTA_count_log(m_sid, 1);
    //KAIN :printf("GPGPU-Sim uArch: core:%3d, cta:%2u initialized @(%lld,%lld)\n", m_sid, free_cta_hw_id, gpu_sim_cycle, gpu_tot_sim_cycle );
}

///////////////////////////////////////////////////////////////////////////////////////////

void dram_t::dram_log( int task ) 
{
   if (task == SAMPLELOG) {
      StatAddSample(mrqq_Dist, que_length());   
   } else if (task == DUMPLOG) {
      printf ("Queue Length DRAM[%d] ",id);StatDisp(mrqq_Dist);
   }
}

//Find next clock domain and increment its time
int gpgpu_sim::next_clock_domain(void) 
{
   double smallest = min3(core_time,icnt_time,dram_time);
   int mask = 0x00;
   if ( l2_time <= smallest ) {
      smallest = l2_time;
      mask |= L2 ;
      l2_time += m_config.l2_period;
   }
   if ( icnt_time <= smallest ) {
      mask |= ICNT;
      icnt_time += m_config.icnt_period;
   }
   if ( dram_time <= smallest ) {
      mask |= DRAM;
      dram_time += m_config.dram_period;
   }
   if ( core_time <= smallest ) {
      mask |= CORE;
      core_time += m_config.core_period;
   }
   return mask;
}

void gpgpu_sim::issue_block2core()
{
    for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
        unsigned idx = (i+m_last_cluster_issue+1) % m_shader_config->n_simt_clusters;
        unsigned num = m_cluster[idx]->issue_block2core();
        if( num ) {
            m_last_cluster_issue=idx;
            m_total_cta_launched += num;
        }
    }
}

unsigned long long g_single_step=0; // set this in gdb to single step the pipeline





///////////////////KAIN
    struct timeval begin;
    struct timeval end; 

    void KAIN_set_begin(void)
    {    
        gettimeofday(&begin,NULL);     
    }    
    void KAIN_set_end(void)
    {    
        gettimeofday(&end,NULL);       
    }    
    double KAIN_time(void)
    {    
        long timeuse = 1000000*(end.tv_sec-begin.tv_sec)+end.tv_usec-begin.tv_usec;     
        double time = timeuse / 1000000.0;
        return time;
    }    
    double time_core1;
    double time_icnt1;
    double time_dram;
    double time_l2;
    double time_icnt2;
    double time_core2;
    double time_core_issue;
	double time_cluster_thread0[20];
	double time_last_thread;
    extern double wb_time[20];
    extern double ex_time[20];
    extern double read_time[20];
    extern double issue_time[20];
    extern double decode_time[20];
    extern double fetch_time[20];

/////////////////////KAIN




    struct timeval begin0[20];
    struct timeval end0[20]; 

    void KAIN_set_begin0(int ID)
    {    
        gettimeofday(begin0+ID,NULL);     
    }    
    void KAIN_set_end0(int ID)
    {    
        gettimeofday(end0+ID,NULL);       
    }    
    double KAIN_time0(int ID)
    {    
        long timeuse = 1000000*(end0[ID].tv_sec-begin0[ID].tv_sec)+end0[ID].tv_usec-begin0[ID].tv_usec;     
        double time = timeuse / 1000000.0;
        return time;
    }    





/////////ADD by KAIN
struct KAIN_SM_run_CTA_str
{
    int ID;
    int begin;
    int end;//KAIN, cluster[begin]--cluster[begin+1]--cluster[end] simulated
};
int KAIN_pthread_init;//ADD by KAIN,to indicate begin
pthread_barrier_t Barrier_KAIN;
int Sem_KAIN_cycleBegin[10];//1 means that you cannot begin/must wait, 0 means ok
int Sem_KAIN_cycleOver[10];//1 means that cycle not over, 0 means ok
pthread_mutex_t KAIN_mutex;//ADD by KAIN to see lock
pthread_mutex_t KAIN_instruction_mutex;
pthread_mutex_t kernel_cores_running;
void *KAIN_Cluster(void *thread_tmp)//Add by KAIN to run cta
{
    int mm; 
    extern gpgpu_sim *g_the_gpu;
	extern int Process_id;
    struct KAIN_SM_run_CTA_str *tmp = (struct KAIN_SM_run_CTA_str *)thread_tmp;
    int ID = tmp->ID;
    int begin = tmp->begin;
    int end = tmp->end;
    printf("process_id is %d,end is %d\n",Process_id,end);

	//////////////////set the CPU affinity
/*	
	cpu_set_t cpuset;
	CPU_SET(61-ID,&cpuset);//0-31
	
	int s = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (s != 0)
	{
        printf("Eror, %d,pthread_setaffinity_np\n",31-ID);
		assert(0);
	}
*/	
	
    while(1)
    {
        //printf("Cluster wait for wake , %d\n",ID);
        while(1)
		{

			if(true == __sync_bool_compare_and_swap(ID+Sem_KAIN_cycleBegin,0,1))
			{
				break;
			}
			else
			{
				if(KAIN_pthread_init == 0)	
					pthread_exit(NULL);
			}

		/*	
			mm++;
			if(mm % 100000 == 0)
            	printf("++++++++++ID %d, Sem Begin is %d\n",ID, *(ID+Sem_KAIN_cycleBegin));
		*/		
		//	printf("wait ID is %d\n",ID);	
		}
		
       // printf("Cluster after wake , %d\n",ID);
					KAIN_set_begin0(ID);

//        printf("ID:%d after wati begin,begin is %d,end is %d\n",ID,begin,end);
        g_the_gpu->run_KAIN_Cluster(begin,end);

 //       printf("after run cluster cycle\n");
 //       printf("Cluster said i after woark, %d\n",ID);
	//	while(1) 
	    	//if(true == __sync_bool_compare_and_swap(ID+Sem_KAIN_cycleOver,1,0));
			//	break;
	    	assert(true==__sync_bool_compare_and_swap(ID+Sem_KAIN_cycleOver,1,0));

					KAIN_set_end0(ID);	
					time_cluster_thread0[ID] += KAIN_time0(ID);

    }
}
void gpgpu_sim::run_KAIN_Cluster(int begin,int end)
{
    for (int i=begin;i<=end;i++) {
       if (m_cluster[i]->get_not_completed() || get_more_cta_left() ) {
          //   printf("cluster ID %d run cycle\n",i);
             m_cluster[i]->core_cycle();
           //  printf("cluster ID %d run cycle  over \n",i);
       }
    }
}



void *Rubbish_recycle(void *rubbishID);
int End[Cluster_Thread_Num];
int BEGIN[Cluster_Thread_Num];

pthread_t thread_id_cluster[Cluster_Thread_Num];








void gpgpu_sim::cycle()
{

    static int init_time_kain = 0;
    if(init_time_kain == 0)
    {

        time_core1 = 0.0;
        time_icnt1 = 0.0;
        time_dram = 0.0;
        time_l2 = 0.0;
        time_icnt2 = 0.0;
        time_core2 = 0.0;
		time_last_thread = 0.0;
        init_time_kain = 1;
		for(int i =0; i < Cluster_Thread_Num; i++)
		{
			time_cluster_thread0[i] = 0.0;
			wb_time[i] = 0.0;
			ex_time[i] = 0.0;
			read_time[i] = 0.0;
			issue_time[i] = 0.0;
			decode_time[i] = 0.0;
			fetch_time[i] = 0.0;
		}
    }
/////////////////////////////////////Test time
   int clock_mask = next_clock_domain();
//////////////////////////////////////////////////?INIT thread
//   printf("int cycle\n"); 
    struct KAIN_SM_run_CTA_str *thread_tmp;//=(struct KAIN_SM_run_CTA_str *)malloc(sizeof(struct KAIN_SM_run_CTA_str)*Cluster_Thread_Num);
	static int init_lock;//this is used to init the mutex for the first time
      if(KAIN_pthread_init == 0)
      {

			KAIN_pthread_init = 1;
			init_lock = 1;

            thread_tmp = (struct KAIN_SM_run_CTA_str *)malloc(sizeof(struct KAIN_SM_run_CTA_str)*Cluster_Thread_Num);
            printf("thread_tmp is %lx\n",thread_tmp);
			fflush(stdout);
           // pthread_barrier_init(&Barrier_KAIN,NULL,Cluster_Thread_Num);


            for(int i = 0; i < Cluster_Thread_Num; i++)
            {
                *(i+Sem_KAIN_cycleOver) = 0;
                *(i+Sem_KAIN_cycleBegin) = 1;
            }
			if(init_lock == 0)
			{
				pthread_mutex_init(&KAIN_mutex,NULL);
				pthread_mutex_init(&KAIN_instruction_mutex,NULL);
				pthread_mutex_init(&kernel_cores_running,NULL);
			}

               //////////////////////////set the low priority of the Producer
               pthread_attr_t attr_P;
               struct sched_param param_P;
               pthread_attr_init(&attr_P);
               pthread_attr_setinheritsched (&attr_P,PTHREAD_EXPLICIT_SCHED);
               pthread_attr_setschedpolicy (&attr_P, SCHED_FIFO);
               param_P.sched_priority = 99; 
               pthread_attr_setschedparam (&attr_P, &param_P);
               //////////////

            for(unsigned i = 0; i < Cluster_Thread_Num;i++)
            {
                int len;
                thread_tmp[i].ID = i;
                if(i < m_shader_config->n_simt_clusters%Cluster_Thread_Num)
                {
                    len = m_shader_config->n_simt_clusters/Cluster_Thread_Num + 1;
                    if (i ==  0)
                    {
                        thread_tmp[i].begin = 0;
                        thread_tmp[i].end = thread_tmp[i].begin+len-1;
                    }
                    else
                    {
                        thread_tmp[i].begin = thread_tmp[i-1].end+1;
                        thread_tmp[i].end = thread_tmp[i].begin+len - 1;
                    }
                }
                else
                {
                    len = m_shader_config->n_simt_clusters/Cluster_Thread_Num;
                    if(i == 0)//KAIN, if Thread_Num = 1
                    {
                        thread_tmp[i].begin = 0;
                        thread_tmp[i].end = thread_tmp[i].begin+len-1;
                    }
                    else
                    {
                        thread_tmp[i].begin = thread_tmp[i-1].end+1;
                        thread_tmp[i].end = thread_tmp[i].begin+len - 1;
                    }
                }
                for(int cl = thread_tmp[i].begin; cl<= thread_tmp[i].end;cl++)
                {
                    m_cluster[cl]->KAINsetThreadID(i);
                }
                End[i] = thread_tmp[i].end;
                BEGIN[i] = thread_tmp[i].begin;

				printf("Before gpgpusim create thread\n");



/*
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            int id = 31-i;
            CPU_SET(id,&cpuset);//0-31
     
            int s = pthread_attr_setaffinity_np(&attr_P, sizeof(cpu_set_t), &cpuset);
            if(s < 0) 
            {    
                printf("error\n");
                assert(0);
            }  

*/



                pthread_create(thread_id_cluster+i,&attr_P,KAIN_Cluster,thread_tmp+i);


            }
	                           /////////////////////////////////////////KAIN set somthing for Cluster delete////////////////////////////////
            //////////////////////////KAIN set the high priority of the Rubbish recycle 
			/*
            pthread_attr_t attr_P_R;
            struct sched_param param_P_R;
            pthread_attr_init(&attr_P_R);
            pthread_attr_setinheritsched (&attr_P_R,PTHREAD_EXPLICIT_SCHED);
            pthread_attr_setschedpolicy (&attr_P_R, SCHED_FIFO);
            param_P_R.sched_priority = 99;
            pthread_attr_setschedparam (&attr_P_R, &param_P_R);
            //////////////
            static int Rubbish_ID[KAIN_rubbish_thread];
            pthread_t thread_id_rubbish[KAIN_rubbish_thread];
            for(int i = 0; i < KAIN_rubbish_thread; i++)
            {
            	Rubbish_ID[i] = i;
            	pthread_create(thread_id_rubbish+i,&attr_P_R,Rubbish_recycle,Rubbish_ID+i);

			}
			*/
            /////////////////////////////////////////////////////////////////////////////////////////////// 

						//////////////////set the CPU affinity
						/*
						cpu_set_t cpuset;
						CPU_ZERO(&cpuset);
						CPU_SET(0,&cpuset);//0-31
						int s = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
						if (s != 0)
						{    
							printf("Eror, %d,pthread_setaffinity_np\n",0);
							assert(0);
						}  
						*/
      }


	KAIN_set_begin();
   if (clock_mask & CORE ) {


       // shader core loading (pop from ICNT into core) follows CORE clock
      int Num = 0;
	  
      for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) 
      {
         m_cluster[i]->icnt_cycle(); 
        // printf("thread_tmp is %lx\n",thread_tmp);
         //printf("i is %d,Num is %d,end is %d\n",i,Num,End[Num]);
		 /*
         if(i >= End[Num])
         {
			__sync_fetch_and_add ((Num+Sem_KAIN_cycleOver),1);// wait for cluster:Num cycle over
            __sync_fetch_and_sub((Num+Sem_KAIN_cycleBegin),1);//cluster:Num cycle now
         //   printf("Cluter you can get woark %d, Sem Begin is %d\n",Num, *(Num+Sem_KAIN_cycleBegin));
            Num++;
         }
		 */
      }
	 
   }
	KAIN_set_end();
	time_core1 += KAIN_time();

	KAIN_set_begin();
    if (clock_mask & ICNT) {
        // pop from memory controller to interconnect
        for (unsigned i=0;i<m_memory_config->m_n_mem;i++) {
            mem_fetch* mf = m_memory_partition_unit[i]->top();
            if (mf) {
                unsigned response_size = mf->get_is_write()?mf->get_ctrl_size():mf->size();
                if ( ::icnt_has_buffer( m_shader_config->mem2device(i), response_size ) ) {
                    if (!mf->get_is_write()) 
                       mf->set_return_timestamp(gpu_sim_cycle+gpu_tot_sim_cycle);
                    mf->set_status(IN_ICNT_TO_SHADER,gpu_sim_cycle+gpu_tot_sim_cycle);
                    ::icnt_push( m_shader_config->mem2device(i), mf->get_tpc(), mf, response_size );
                    m_memory_partition_unit[i]->pop();
                } else {
                    gpu_stall_icnt2sh++;
                }
            } else {
               m_memory_partition_unit[i]->pop();
            }
        }
    }
KAIN_set_end();
 time_icnt1 += KAIN_time();


KAIN_set_begin();
   if (clock_mask & DRAM) {
      for (unsigned i=0;i<m_memory_config->m_n_mem;i++)  
         m_memory_partition_unit[i]->dram_cycle(); // Issue the dram command (scheduler + delay model) 
   }
KAIN_set_end();
 time_dram += KAIN_time();
   // L2 operations follow L2 clock domain
KAIN_set_begin();
   if (clock_mask & L2) {
      for (unsigned i=0;i<m_memory_config->m_n_mem;i++) {
          //move memory request from interconnect into memory partition (if not backed up)
          //Note:This needs to be called in DRAM clock domain if there is no L2 cache in the system
          if ( m_memory_partition_unit[i]->full() ) {
             gpu_stall_dramfull++;
          } else {
              mem_fetch* mf = (mem_fetch*) icnt_pop( m_shader_config->mem2device(i) );
              m_memory_partition_unit[i]->push( mf, gpu_sim_cycle + gpu_tot_sim_cycle );
          }
          m_memory_partition_unit[i]->cache_cycle(gpu_sim_cycle+gpu_tot_sim_cycle);
      }
   }
KAIN_set_end();
 time_l2 += KAIN_time();



KAIN_set_begin();
   if (clock_mask & ICNT) {
      icnt_transfer();
   }
KAIN_set_end();
 time_icnt2 += KAIN_time();


   if (clock_mask & CORE) {
	   /*
	   if(gpu_sim_cycle>149000)
	   {
		   printf("begin core cycle,sleep\n");
		   sleep(3600);
		   fflush(stdout);
	   }
	   */

KAIN_set_begin();


      // L1 cache + shader core pipeline stages 
/*
      for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
         if (m_cluster[i]->get_not_completed() || get_more_cta_left() ) {
               m_cluster[i]->core_cycle();
         }
      }
*/

         for(int Num = 0; Num < Cluster_Thread_Num; Num++)
         {
			__sync_fetch_and_add ((Num+Sem_KAIN_cycleOver),1);// wait for cluster:Num cycle over
            __sync_fetch_and_sub((Num+Sem_KAIN_cycleBegin),1);//cluster:Num cycle now
         //   printf("Cluter you can get woark %d, Sem Begin is %d\n",Num, *(Num+Sem_KAIN_cycleBegin));
         }

        unsigned long long count = 0;
		int mask_complete[10]={0,0,0,0,0,0,0,0,0,0};
		int error_count = 0;
        for(unsigned i = 0; 1;i++)
        {
			if(i == Cluster_Thread_Num)
				i = 0;
            //printf("before wait CycleOver %d\n",i);
            if(__sync_bool_compare_and_swap(i+Sem_KAIN_cycleOver,0,0) == true && mask_complete[i] != 1)
			{
				if(count == 0)
				{
					KAIN_set_end();
					time_last_thread+=KAIN_time();
				}
				mask_complete[i]  = 1;
           // 	printf("CycleOver %d\n",i);
				count++;	
			}
			if(count == Cluster_Thread_Num)
				break;
								/*
								int begin = BEGIN[i];
								int end = End[i];
								for(int j = begin; j<=end;j++)
								{
								   for(std::list<mem_fetch *>::iterator plist = m_cluster[j]->m_KAIN_process.begin();plist !=m_cluster[j]->m_KAIN_process.end();plist++)
								   {
									   mem_fetch *mf = *plist;
									   m_cluster[j]->icnt_inject_request_packet(mf);
								   }
									   m_cluster[j]->m_KAIN_process.clear();
								}
								*/
        }
KAIN_set_end();
 time_core2 += KAIN_time();

 /*
	   if(gpu_sim_cycle>149000)
	   {
		   printf("after core cycle\n");
		   fflush(stdout);
	   }
	   */
/*
        for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) 
        {
            printf("cluster ID: %d before wati cycle over\n",i);
            sem_wait(i+Sem_KAIN_cycleOver);   
            for(std::list<mem_fetch *>::iterator plist = m_cluster[i]->m_KAIN_process.begin();plist !=m_cluster[i]->m_KAIN_process.end();plist++)
            {
                mem_fetch *mf = *plist;
                m_cluster[i]->icnt_inject_request_packet(mf);
            }
            m_cluster[i]->m_KAIN_process.clear();

            printf("Cluster ID: %d,after wati cycle over\n",i);
        }
*/


     // printf("out core cycle\n");
      if( g_single_step && ((gpu_sim_cycle+gpu_tot_sim_cycle) >= g_single_step) ) {
          asm("int $03");
      }
      gpu_sim_cycle++;
      if( g_interactive_debugger_enabled ) 
         gpgpu_debug();
     
      issue_block2core();
      
      // Flush the caches once all of threads are completed.
      if (m_config.gpgpu_flush_cache) {
         int all_threads_complete = 1 ; 
         for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
            if (m_cluster[i]->get_not_completed() == 0) 
               m_cluster[i]->cache_flush();
            else 
               all_threads_complete = 0 ; 
         }
         if (all_threads_complete && !m_memory_config->m_L2_config.disabled() ) {
            printf("Flushed L2 caches...\n");
            if (m_memory_config->m_L2_config.get_num_lines()) {
               int dlc = 0;
               for (unsigned i=0;i<m_memory_config->m_n_mem;i++) {
                  dlc = m_memory_partition_unit[i]->flushL2();
                  assert (dlc == 0); // need to model actual writes to DRAM here
                  printf("Dirty lines flushed from L2 %d is %d\n", i, dlc  );
               }
            }
         }
      }

      if (!(gpu_sim_cycle % m_config.gpu_stat_sample_freq)) {
         time_t days, hrs, minutes, sec;
         time_t curr_time;
         time(&curr_time);
         unsigned long long  elapsed_time = MAX(curr_time - g_simulation_starttime, 1);
         days    = elapsed_time/(3600*24);
         hrs     = elapsed_time/3600 - 24*days;
         minutes = elapsed_time/60 - 60*(hrs + 24*days);
         sec = elapsed_time - 60*(minutes + 60*(hrs + 24*days));
		
	
		 //if(Process_id == 0)
         printf("GPGPU-Sim uArch: cycles simulated: %lld  inst.: %lld (ipc=%4.1f) sim_rate=%u (inst/sec) elapsed = %u:%u:%02u:%02u / %s", 
                gpu_tot_sim_cycle + gpu_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn, 
                (double)gpu_sim_insn/(double)gpu_sim_cycle,
                (unsigned)((gpu_tot_sim_insn+gpu_sim_insn) / elapsed_time),
                (unsigned)days,(unsigned)hrs,(unsigned)minutes,(unsigned)sec,
                ctime(&curr_time));

         fflush(stdout);
		 
         m_memory_stats->memlatstat_lat_pw();
         visualizer_printstat();
         if (m_config.gpgpu_runtime_stat && (m_config.gpu_runtime_stat_flag != 0) ) {
            if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
               for (unsigned i=0;i<m_memory_config->m_n_mem;i++) 
                  m_memory_partition_unit[i]->print_stat(stdout);
               printf("maxmrqlatency = %d \n", m_memory_stats->max_mrq_latency);
               printf("maxmflatency = %d \n", m_memory_stats->max_mf_latency);
            }
            if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SHD_INFO) 
               shader_print_runtime_stat( stdout );
            if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_L1MISS) 
               shader_print_l1_miss_stat( stdout );
         }
      }

      if (!(gpu_sim_cycle % 20000)) {
         // deadlock detection 
         if (m_config.gpu_deadlock_detect && gpu_sim_insn == last_gpu_sim_insn) {
            gpu_deadlock = true;//KAIN delet this, beacuase gpu_sim_insn is the multi thread compete
         } else {
            last_gpu_sim_insn = gpu_sim_insn;
         }
      }
      try_snap_shot(gpu_sim_cycle);
      spill_log_to_file (stdout, 0, gpu_sim_cycle);
   }
   //printf("core2 over\n");
}

void shader_core_ctx::dump_warp_state( FILE *fout ) const
{
   fprintf(fout, "\n");
   fprintf(fout, "per warp functional simulation status:\n");
   for (unsigned w=0; w < m_config->max_warps_per_shader; w++ ) 
       m_warp[w].print(fout);
}

void gpgpu_sim::dump_pipeline( int mask, int s, int m ) const
{
/*
   You may want to use this function while running GPGPU-Sim in gdb.
   One way to do that is add the following to your .gdbinit file:
 
      define dp
         call g_the_gpu.dump_pipeline_impl((0x40|0x4|0x1),$arg0,0)
      end
 
   Then, typing "dp 3" will show the contents of the pipeline for shader core 3.
*/

   printf("Dumping pipeline state...\n");
   if(!mask) mask = 0xFFFFFFFF;
   for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
      if(s != -1) {
         i = s;
      }
      if(mask&1) m_cluster[m_shader_config->sid_to_cluster(i)]->display_pipeline(i,stdout,1,mask & 0x2E);
      if(s != -1) {
         break;
      }
   }
   if(mask&0x10000) {
      for (unsigned i=0;i<m_memory_config->m_n_mem;i++) {
         if(m != -1) {
            i=m;
         }
         printf("DRAM / memory controller %u:\n", i);
         if(mask&0x100000) m_memory_partition_unit[i]->print_stat(stdout);
         if(mask&0x1000000)   m_memory_partition_unit[i]->visualize();
         if(mask&0x10000000)   m_memory_partition_unit[i]->print(stdout);
         if(m != -1) {
            break;
         }
      }
   }
   fflush(stdout);
}

const struct shader_core_config * gpgpu_sim::getShaderCoreConfig()
{
   return m_shader_config;
}

const struct memory_config * gpgpu_sim::getMemoryConfig()
{
   return m_memory_config;
}

simt_core_cluster * gpgpu_sim::getSIMTCluster()
{
   return *m_cluster;
}

void memory_partition_unit::visualizer_print( gzFile visualizer_file )
{
   m_dram->visualizer_print(visualizer_file);
}

