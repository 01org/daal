/* file: sgd_dense_minibatch_oneapi_impl.i */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of SGD dense minibatch Batch algorithm on GPU.
//--
*/

#include "algorithms/kernel/optimization_solver/sgd/oneapi/cl_kernel/sgd_dense_minibatch.cl"
#include "algorithms/kernel/optimization_solver/iterative_solver_kernel.h"
#include "data_management/data/numeric_table_sycl_homogen.h"
#include "externals/service_math.h"

#include "externals/service_ittnotify.h"

DAAL_ITTNOTIFY_DOMAIN(optimization_solver.sgd.batch.oneapi);

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sgd
{
namespace internal
{
using namespace daal::oneapi::internal;
using namespace daal::data_management;

static uint32_t getWorkgroupsCount(const uint32_t n, const uint32_t localWorkSize)
{
    const uint32_t elementsPerGroup = localWorkSize;
    uint32_t workgroupsCount        = n / elementsPerGroup;

    if (workgroupsCount * elementsPerGroup < n)
    {
        workgroupsCount++;
    }
    return workgroupsCount;
}

template <typename algorithmFPType, CpuType cpu>
services::Status SGDKernelOneAPI<algorithmFPType, miniBatch, cpu>::makeStep(const uint32_t argumentSize,
                                                                            const services::Buffer<algorithmFPType> & prevWorkValueBuff,
                                                                            const services::Buffer<algorithmFPType> & gradientBuff,
                                                                            services::Buffer<algorithmFPType> & workValueBuff,
                                                                            const algorithmFPType learningRate, const algorithmFPType consCoeff)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(makeStep);
    services::Status status;

    ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    buildProgram(factory);

    const char * const kernelName = "makeStep";
    KernelPtr kernel              = factory.getKernel(kernelName);

    KernelArguments args(5);
    args.set(0, gradientBuff, AccessModeIds::read);
    args.set(1, prevWorkValueBuff, AccessModeIds::read);
    args.set(2, workValueBuff, AccessModeIds::readwrite);
    args.set(3, learningRate);
    args.set(4, consCoeff);

    KernelRange range(argumentSize);
    ctx.run(range, kernel, args, &status);

    return status;
}

template <typename algorithmFPType, CpuType cpu>
static services::Status sumReduction(const services::Buffer<algorithmFPType> & reductionBuffer, const size_t nWorkGroups, algorithmFPType & result)
{
    auto sumReductionArrayPtr      = reductionBuffer.toHost(data_management::readOnly);
    const auto * sumReductionArray = sumReductionArrayPtr.get();

    // Final summation with CPU
    for (size_t i = 0; i < nWorkGroups; i++)
    {
        result += sumReductionArray[i];
    }
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status SGDKernelOneAPI<algorithmFPType, miniBatch, cpu>::vectorNorm(const services::Buffer<algorithmFPType> & x, const uint32_t n,
                                                                              algorithmFPType & norm)
{
    services::Status status;

    const TypeIds::Id idType = TypeIds::id<algorithmFPType>();

    ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    buildProgram(factory);

    const char * const kernelName = "sumSq";
    KernelPtr kernel              = factory.getKernel(kernelName);

    // InfoDevice& info = ctx.getInfoDevice();
    // const size_t maxWorkItemSizes1d = info.max_work_item_sizes_1d;
    // const size_t maxWorkGroupSize = info.max_work_group_size;

    size_t workItemsPerGroup = 256;
    const size_t nWorkGroups = getWorkgroupsCount(n, workItemsPerGroup);

    KernelRange localRange(workItemsPerGroup);
    KernelRange globalRange(workItemsPerGroup * nWorkGroups);

    KernelNDRange range(1);

    range.local(localRange, &status);
    range.global(globalRange, &status);
    DAAL_CHECK_STATUS_VAR(status);

    UniversalBuffer buffer                            = ctx.allocate(idType, nWorkGroups, &status);
    services::Buffer<algorithmFPType> reductionBuffer = buffer.get<algorithmFPType>();

    KernelArguments args(3 /*4*/);
    args.set(0, x, AccessModeIds::read);
    args.set(1, n);
    args.set(2, reductionBuffer, AccessModeIds::write);
    //args.set(3, LocalBuffer(idType, workItemsPerGroup));

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(vectorNorm.run);
        ctx.run(range, kernel, args, &status);
    }

    status = sumReduction<algorithmFPType, cpu>(reductionBuffer, nWorkGroups, norm);

    norm = daal::internal::Math<algorithmFPType, cpu>::sSqrt(norm);

    return status;
}

template <typename algorithmFPType, CpuType cpu>
void SGDKernelOneAPI<algorithmFPType, miniBatch, cpu>::buildProgram(ClKernelFactoryIface & factory)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(buildProgram);
    services::String options = getKeyFPType<algorithmFPType>();

    services::String cachekey("__daal_algorithms_optimization_solver_sgd_");
    cachekey.add(options);
    options.add(" -D LOCAL_SUM_SIZE=256 ");

    factory.build(ExecutionTargetIds::device, cachekey.c_str(), clKernelSGDMiniBatch, options.c_str());
}

template <typename algorithmFPType, CpuType cpu>
services::Status SGDKernelOneAPI<algorithmFPType, miniBatch, cpu>::compute(HostAppIface * pHost, NumericTable * inputArgument,
                                                                           NumericTablePtr minimum, NumericTable * nIterations,
                                                                           Parameter<miniBatch> * parameter, NumericTable * learningRateSequence,
                                                                           NumericTable * batchIndices, OptionalArgument * optionalArgument,
                                                                           OptionalArgument * optionalResult, engines::BatchBase & engine)
{
    printf("SGDKernelOneAPI<algorithmFPType, miniBatch, cpu>::compute\n");

    services::Status status;

    ExecutionContextIface & ctx = services::Environment::getInstance()->getDefaultExecutionContext();

    const size_t argumentSize = inputArgument->getNumberOfRows();
    const size_t nIter        = parameter->nIterations;
    const size_t L            = parameter->innerNIterations;
    const size_t batchSize    = parameter->batchSize;

    WriteRows<int, cpu> nIterationsBD(*nIterations, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(nIterationsBD);
    int * nProceededIterations = nIterationsBD.get();
    // if nIter == 0, set result as start point, the number of executed iters to 0
    if (nIter == 0 || L == 0)
    {
        nProceededIterations[0] = 0;
        return status;
    }

    NumericTable * lastIterationInput = optionalArgument ? NumericTable::cast(optionalArgument->get(iterative_solver::lastIteration)).get() : nullptr;
    NumericTable * pastWorkValueInput = optionalArgument ? NumericTable::cast(optionalArgument->get(sgd::pastWorkValue)).get() : nullptr;

    NumericTable * lastIterationResult = optionalResult ? NumericTable::cast(optionalResult->get(iterative_solver::lastIteration)).get() : nullptr;
    NumericTable * pastWorkValueResult = optionalResult ? NumericTable::cast(optionalResult->get(sgd::pastWorkValue)).get() : nullptr;

    const double accuracyThreshold = parameter->accuracyThreshold;

    sum_of_functions::BatchPtr function = parameter->function;
    const size_t nTerms                 = function->sumOfFunctionsParameter->numberOfTerms;

    DAAL_ASSERT(minimum->getNumberOfRows() == argumentSize);

    BlockDescriptor<algorithmFPType> workValueBD;
    DAAL_CHECK_STATUS(status, minimum->getBlockOfRows(0, argumentSize, ReadWriteMode::readWrite, workValueBD));
    services::Buffer<algorithmFPType> workValueBuff = workValueBD.getBuffer();

    auto workValueSNT = SyclHomogenNumericTable<algorithmFPType>::create(workValueBuff, 1, argumentSize);

    NumericTablePtr previousArgument = function->sumOfFunctionsInput->get(sum_of_functions::argument);
    function->sumOfFunctionsInput->set(sum_of_functions::argument, workValueSNT);

    ReadRows<algorithmFPType, cpu> learningRateBD(*learningRateSequence, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(learningRateBD);
    const algorithmFPType * const learningRateArray = learningRateBD.get();

    NumericTable * conservativeSequence = parameter->conservativeSequence.get();
    ReadRows<algorithmFPType, cpu> consCoeffsBD(*conservativeSequence, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(consCoeffsBD);
    const algorithmFPType * const consCoeffsArray = consCoeffsBD.get();

    const size_t consCoeffsLength   = conservativeSequence->getNumberOfColumns();
    const size_t learningRateLength = learningRateSequence->getNumberOfColumns();

    const IndicesStatus indicesStatus = (batchIndices ? user : (batchSize < nTerms ? random : all));
    services::SharedPtr<HomogenNumericTableCPU<int, cpu> > ntBatchIndices;

    // if (indicesStatus == user || indicesStatus == random)
    // {
    //     // Replace by SyclNumericTable when will be RNG on GPU
    //     ntBatchIndices = HomogenNumericTableCPU<int, cpu>::create(batchSize, 1, &status);
    // }

    // NumericTablePtr previousBatchIndices            = function->sumOfFunctionsParameter->batchIndices;
    // function->sumOfFunctionsParameter->batchIndices = ntBatchIndices;

    // const TypeIds::Id idType                            = TypeIds::id<algorithmFPType>();
    // UniversalBuffer prevWorkValueU                      = ctx.allocate(idType, argumentSize, &status);
    // services::Buffer<algorithmFPType> prevWorkValueBuff = prevWorkValueU.get<algorithmFPType>();

    // size_t startIteration = 0, nProceededIters = 0;
    // if (lastIterationInput)
    // {
    //     ReadRows<int, cpu, NumericTable> lastIterationInputBD(lastIterationInput, 0, 1);
    //     const int * lastIterationInputArray = lastIterationInputBD.get();
    //     startIteration                      = lastIterationInputArray[0];
    // }

    // if (pastWorkValueInput)
    // {
    //     BlockDescriptor<algorithmFPType> pastWorkValueInputBD;
    //     pastWorkValueInput->getBlockOfRows(0, argumentSize, ReadWriteMode::readOnly, pastWorkValueInputBD);

    //     const services::Buffer<algorithmFPType> pastWorkValueInputBuff = pastWorkValueInputBD.getBuffer();

    //     ctx.copy(prevWorkValueBuff, 0, pastWorkValueInputBuff, 0, argumentSize, &status);
    //     pastWorkValueInput->releaseBlockOfRows(pastWorkValueInputBD);
    // }
    // else
    // {
    //     ctx.fill(prevWorkValueU, 0.0, &status);
    // }

    // // init workValue
    // BlockDescriptor<algorithmFPType> startValueBD;
    // DAAL_CHECK_STATUS(status, inputArgument->getBlockOfRows(0, argumentSize, ReadWriteMode::readOnly, startValueBD));
    // const services::Buffer<algorithmFPType> startValueBuff = startValueBD.getBuffer();
    // ctx.copy(workValueBuff, 0, startValueBuff, 0, argumentSize, &status);

    // DAAL_CHECK_STATUS(status, inputArgument->releaseBlockOfRows(startValueBD));

    // ReadRows<int, cpu> predefinedBatchIndicesBD(batchIndices, 0, nIter);
    // iterative_solver::internal::RngTask<int, cpu> rngTask(predefinedBatchIndicesBD.get(), batchSize);
    // rngTask.init(nTerms, engine);

    // algorithmFPType learningRate = learningRateArray[0];
    // algorithmFPType consCoeff    = consCoeffsArray[0];

    // UniversalBuffer gradientU                      = ctx.allocate(idType, argumentSize, &status);
    // services::Buffer<algorithmFPType> gradientBuff = gradientU.get<algorithmFPType>();

    // auto gradientSNT = SyclHomogenNumericTable<algorithmFPType>::create(gradientBuff, 1, argumentSize);
    // function->getResult()->set(objective_function::gradientIdx, gradientSNT);

    // *nProceededIterations = static_cast<int>(nIter);

    // services::internal::HostAppHelper host(pHost, 10);
    // for (size_t epoch = startIteration; epoch < (startIteration + nIter); epoch++)
    // {
    //     if (epoch % L == 0 || epoch == startIteration)
    //     {
    //         learningRate = learningRateArray[(epoch / L) % learningRateLength];
    //         consCoeff    = consCoeffsArray[(epoch / L) % consCoeffsLength];
    //         if (indicesStatus == user || indicesStatus == random)
    //         {
    //             DAAL_ITTNOTIFY_SCOPED_TASK(generateUniform);
    //             const int * pValues = nullptr;
    //             DAAL_CHECK_STATUS(status, rngTask.get(pValues));
    //             ntBatchIndices->setArray(const_cast<int *>(pValues), ntBatchIndices->getNumberOfRows());
    //         }
    //     }

    //     DAAL_CHECK_STATUS(status, function->computeNoThrow());

    //     if (host.isCancelled(status, 1))
    //     {
    //         *nProceededIterations = static_cast<int>(epoch - startIteration);
    //         break;
    //     }

    //     if (epoch % L == 0)
    //     {
    //         if (nIter > 1 && accuracyThreshold > 0)
    //         {
    //             algorithmFPType pointNorm = algorithmFPType(0), gradientNorm = algorithmFPType(0);
    //             vectorNorm(workValueBuff, argumentSize, pointNorm);
    //             vectorNorm(gradientBuff, argumentSize, gradientNorm);
    //             const double gradientThreshold = accuracyThreshold * daal::internal::Math<algorithmFPType, cpu>::sMax(1.0, pointNorm);

    //             if (gradientNorm < gradientThreshold)
    //             {
    //                 *nProceededIterations = static_cast<int>(epoch - startIteration);
    //                 break;
    //             }
    //         }

    //         ctx.copy(prevWorkValueBuff, 0, workValueBuff, 0, argumentSize, &status);
    //     }
    //     DAAL_CHECK_STATUS(status, makeStep(argumentSize, prevWorkValueBuff, gradientBuff, workValueBuff, learningRate, consCoeff));
    //     nProceededIters++;
    // }

    // if (lastIterationResult)
    // {
    //     WriteRows<int, cpu, NumericTable> lastIterationResultBD(lastIterationResult, 0, 1);
    //     int * lastIterationResultArray = lastIterationResultBD.get();
    //     lastIterationResultArray[0]    = startIteration + nProceededIters;
    // }

    // if (pastWorkValueResult)
    // {
    //     BlockDescriptor<algorithmFPType> pastWorkValueResultBD;
    //     pastWorkValueResult->getBlockOfRows(0, argumentSize, ReadWriteMode::writeOnly, pastWorkValueResultBD);

    //     services::Buffer<algorithmFPType> pastWorkValueResultBuffer = pastWorkValueResultBD.getBuffer();

    //     ctx.copy(pastWorkValueResultBuffer, 0, prevWorkValueBuff, 0, argumentSize, &status);
    //     pastWorkValueResult->releaseBlockOfRows(pastWorkValueResultBD);
    // }

    // DAAL_CHECK_STATUS(status, minimum->releaseBlockOfRows(workValueBD));

    // function->sumOfFunctionsParameter->batchIndices = previousBatchIndices;
    // function->sumOfFunctionsInput->set(sum_of_functions::argument, previousArgument);
    return status;
}

} // namespace internal
} // namespace sgd
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
