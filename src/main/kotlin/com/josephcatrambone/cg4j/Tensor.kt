package com.josephcatrambone.cg4j

import org.nd4j.linalg.api.ndarray.INDArray

/**
 * Created by jcatrambone on 12/13/16.
 */

class Tensor(var shape: IntArray, var data: FloatArray) {
	private lateinit var dimensionOffset: IntArray // Use this to determine the offset of the item in the linear array

	init {
		dimensionOffset = IntArray(size=shape.size)
		for(index in (0..shape.size-2)) {
			dimensionOffset[index] = shape.slice(index+1..shape.size-1).reduce { a, b -> a*b }
		}
		dimensionOffset[shape.size-1] = 1
	}

	companion object {
		/*
		@JvmStatic fun newFromData(vararg shape: Int, data: FloatArray) {

		}
		*/
	}

	private fun indexArrayToIndex(vararg index: Int): Int {
		// Flatten an array of items into a singular index.
		// (w*y) + x
		// (h*w*z) + (w*y) + x
		// (d*h*w*p) + (h*w*z) + (w*y) + x
		// (b*d*h*w*q) + (d*h*w*p) + (h*w*z) + (w*y) + x
		return index.foldIndexed(0, {index, accumulator, element -> dimensionOffset[index]*element + accumulator})
	}

	private fun indexToIndexArray(index: Int): IntArray {
		// Remap an index into a point array.
		// b, d, h, w -> index
		throw NotImplementedError()
	}

	operator fun get(vararg index: Int): Float {
		assert(index.size == shape.size)
		return data[indexArrayToIndex(*index)]
	}

	operator fun set(vararg index: Int, value: Float) {
		data[indexArrayToIndex(*index)] = value
	}

	fun elementOperation_i(f:(Float)->Float) {
		for(i in (0..data.size)) {
			data[i] = f(data[i])
		}
	}

	fun mmul(rhs: Tensor): Tensor {
		assert(this.shape[this.shape.size-1] == rhs.shape[0])
		var out = Tensor.zeros(*(this.shape.sliceArray(0..this.shape.size-2).plus(rhs.shape.sliceArray(1..rhs.shape.size-1))))

		//for(leftIndex in (0..))
		return out
	}

	fun contract(rhs: Tensor, dimension:Int): Tensor {
		throw NotImplementedError()
	}

	fun slice(vararg slices: IntRange): Tensor {
		assert(slices.size == shape.size)
		throw NotImplementedError()
	}

	fun setSlice(vararg slices: IntRange, value: Tensor) {
		for(index in (0..value.shape.size)) {
			assert(slices[index].endInclusive - slices[index].start == value.shape[index])

			throw NotImplementedError()
		}
	}
}

fun Tensor.Companion.newFromFun(vararg shape: Int, initFunction: (Int)->Float): Tensor {
	return Tensor(shape, FloatArray(init = {index -> initFunction(index)}, size = shape.reduce { a, b -> a*b }))
}

fun Tensor.Companion.zeros(vararg shape: Int): Tensor {
	return Tensor(shape, FloatArray(size=shape.reduce { a, b -> a*b }))
}

fun Tensor.Companion.ones(vararg shape: Int): Tensor {
	return Tensor(shape, FloatArray(size=shape.reduce { a, b -> a*b }, init = { a->1.0f }))
}