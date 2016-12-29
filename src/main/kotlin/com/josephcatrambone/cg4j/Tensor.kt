package com.josephcatrambone.cg4j

import java.util.*

/**
 * Created by jcatrambone on 12/13/16.
 */

class Tensor(var shape: IntArray, var data: FloatArray) {
	private var dimensionOffset: IntArray // Use this to determine the offset of the item in the linear array

	init {
		dimensionOffset = IntArray(size=shape.size)
		for(index in (0..shape.size-2)) {
			dimensionOffset[index] = shape.slice(index+1..shape.size-1).reduce { a, b -> a*b }
		}
		dimensionOffset[shape.size-1] = 1
	}

	companion object {
		val random = Random()
		/*
		@JvmStatic fun newFromData(vararg shape: Int, data: FloatArray) {

		}
		*/
	}

	fun indexArrayToIndex(vararg index: Int): Int {
		// Flatten an array of items into a singular index.
		// (w*y) + x
		// (h*w*z) + (w*y) + x
		// (d*h*w*p) + (h*w*z) + (w*y) + x
		// (b*d*h*w*q) + (d*h*w*p) + (h*w*z) + (w*y) + x
		return index.foldIndexed(0, {index, accumulator, element -> dimensionOffset[index]*element + accumulator})
	}

	fun indexToIndexArray(index: Int): IntArray {
		//return Tensor.indexToCoordinate(this.shape, index)

		// Remap an index into a point array.
		// b, d, h, w -> index
		// Treat the index as a 'count' item and iterate over the indices in order.
		var quantLeft = index
		val indexArray: MutableList<Int> = mutableListOf()
		for(dimensionSize in dimensionOffset) {
			indexArray.add(quantLeft / dimensionSize)
			quantLeft %= dimensionSize
		}
		return indexArray.toIntArray()
	}

	operator fun get(vararg index: Int): Float {
		assert(index.size == shape.size)
		return data[indexArrayToIndex(*index)]
	}

	operator fun set(vararg index: Int, value: Float) {
		data[indexArrayToIndex(*index)] = value
	}

	fun elementOperation(f: (Float)->Float): Tensor {
		val t = dup()
		for(i in (0..data.size-1)) {
			t.data[i] = f(t.data[i])
		}
		return t
	}

	fun elementOperation(rhs: Tensor, f: (Float, Float)->Float): Tensor {
		val t = dup()
		for(i in (0..data.size-1)) {
			t.data[i] = f(t.data[i], rhs.data[i])
		}
		return t
	}

	fun elementOperation_i(f:(Float)->Float) {
		for(i in (0..data.size-1)) {
			data[i] = f(data[i])
		}
	}

	fun elementOperation_i(rhs: Tensor, f:(Float, Float)->Float) {
		for(i in (0..data.size-1)) {
			data[i] = f(data[i], rhs.data[i])
		}
	}

	// Tensor Contraction
	fun mmul(rhs: Tensor): Tensor {
		assert(this.shape[this.shape.size-1] == rhs.shape[0])
		var out = Tensor.zeros(*(this.shape.sliceArray(0..this.shape.size-2).plus(rhs.shape.sliceArray(1..rhs.shape.size-1))))
		// We need to 'count up' through all the dimensions except the last for the outer tensor.
		// We need to 'count up' through all the dimensions except the FIRST for the inner tensor.
		// We need to count in parallel through the dimensions and accumulate the product for the inner-most dimension.
		//
		// Cijmno = for all k, Aijk * Bkmno
		//

		// Build the dimension iterator for the left hand side, save the last dimension.
		/*
		val lhsDimensionOffset = this.dimensionOffset.slice(0..this.dimensionOffset.size-2)
		val lhsDimensionSize = this.shape.reduce({a, b -> a*b}) / this.shape.last() // [A, B, C, D] -> A*B*C
		val rhsDimensionOffset = rhs.dimensionOffset.slice(1..rhs.dimensionOffset.size-1)
		val rhsDimensionSize = rhs.shape.reduce({a, b -> a*b}) / rhs.shape[0]

		for(i in (0..lhsDimensionSize-1)) {
			for(j in (0..rhsDimensionSize-1)) {
				// Calculate the position of the three left-most points.
				val leftIndexArray = lhsDimensionOffset.map()
				var accumulator = 0f
				for(k in (0..this.shape.last()-1)) {

				}
			}
		}
		*/

		for(i in (0..this.data.size-1)) {
			for(j in (0..rhs.data.size-1)) {
				val lhsIndex = this.indexToIndexArray(i)
				val rhsIndex = rhs.indexToIndexArray(j)
				if(lhsIndex.last() == rhsIndex.first()) {
					val outIndex = mutableListOf<Int>()
					lhsIndex.dropLast(1).toCollection(outIndex)
					rhsIndex.drop(1).toCollection(outIndex)
					val expandedOutIndex : IntArray = outIndex.toIntArray()
					out.set(*expandedOutIndex, value=out.get(*expandedOutIndex)+(this.data[i]*rhs.data[j]))
				}
			}
		}

		//for(leftIndex in (0..))
		return out
	}

	/***
	 * Like slice, but won't produce a tensor with the same dimensions as the parent.
	 * Instead, will return the subtensor.  I.e. if you have [5, 4, 3, 2], a call to
	 * getSubtensor(0, 4) will give back a tensor of shape [4, 3, 2] from the tensor at [4, :, :, :],
	 * getSubtensor(2, 2) will give back a tensor of shape [5, 4, 2] from the tensor at [:, :, 2, :]
	 */
	fun getSubtensor(axis: Int, axisIndex: Int): Tensor {
		// Select all the dimensions EXCEPT the one specified by axis.
		val out: Tensor = Tensor.zeros(
			*this.shape.filterIndexed{axisIndex, shapeValue -> axisIndex != axis}.toIntArray()
		)

		// Go through the values here and copy the items.
		val maxIndex = out.shape.reduce({a, b -> a*b})
		for(index in (0..maxIndex-1)) {
			// Map from index to the position in the out object.
			val outIndexArray = out.indexToIndexArray(index)
			val inIndexArray = IntArray(size=this.shape.size, init = { i -> if(i<axis) {outIndexArray[i]} else if (i == axis) { axisIndex } else {outIndexArray[i-1]} })
			out.set(*outIndexArray, value=this.get(*inIndexArray))
		}

		return out
	}

	fun setSubtensor(axis: Int, axisIndex: Int, value: Tensor) {
		// Go through the values here and copy the items.
		val maxIndex = value.shape.reduce({a, b -> a*b})
		for(index in (0..maxIndex-1)) {
			// Map from index to the position in the out object.
			val vIndexArray = value.indexToIndexArray(index)
			val inIndexArray = IntArray(size=this.shape.size, init = { i -> if(i==axis) {axisIndex} else {vIndexArray[i]} })
			this.set(*inIndexArray, value=value.get(*vIndexArray))
		}
	}

	fun slice(vararg slices: IntRange): Tensor {
		assert(slices.size == shape.size)

		// Off by one.  The slices are inclusive, but we have to allocate +1 to make them match.
		val out = Tensor.zeros(*slices.map { x -> 1+x.last-x.start }.toIntArray())
		val offsets = slices.map{ x -> x.start }.toIntArray()
		val numOutputFloats = out.shape.reduce {a, b -> a*b}

		// Iterate through each of the possible indices, building them up from the outside.
		for(i in (0..numOutputFloats-1)) {
			// Get the position of i in the out tensor.
			val outTensorPosition = out.indexToIndexArray(i)
			val inTensorPosition = outTensorPosition.zip(offsets).map { p -> p.first+p.second }.toIntArray()
			out.set(*outTensorPosition, value=this.get(*inTensorPosition))
		}

		return out
	}

	fun setSlice(vararg slices: IntRange, value: Tensor) {
		assert(slices.size == shape.size)

		val offsets = slices.map{ x -> x.start }.toIntArray()
		val numFloats = value.shape.reduce {a, b -> a*b}

		// Iterate through each of the possible indices, building them up from the outside.
		for(i in (0..numFloats-1)) {
			// Get the position of i in the out tensor.
			val thatTensorPosition = value.indexToIndexArray(i)
			val thisTensorPosition = thatTensorPosition.zip(offsets).map { p -> p.first+p.second }.toIntArray()
			this.set(*thisTensorPosition, value=value.get(*thatTensorPosition))
		}
	}

	fun dup(): Tensor {
		return Tensor(this.shape.copyOf(), data.copyOf())
	}

	fun transpose(): Tensor {
		if(shape.size > 2) {
			throw NotImplementedError("Not yet implemented for greater than 2D matrices.")
		} else if(shape.size == 2) {
			return Tensor.newFromFun(shape[1], shape[0], initFunction = { index: Int -> this.get(index / this.shape[1], index % this.shape[1]) })
		} else if(shape.size == 1) {
			return Tensor.newFromFun(shape[0], 1, initFunction = { index: Int -> this.data[index] })
		} else {
			throw NotImplementedError("Transpose not valid for 0D matrices.")
		}
	}

	override fun toString(): String {
		// Make sure this stays in sync with fromString.
		val sb = StringBuilder()
		sb.append("TENSOR[")
		sb.append(this.shape.joinToString())
		sb.append("] : [")
		sb.append(this.data.joinToString())
		sb.append("]")
		return sb.toString()
	}

	//
	// Algebraic operations.  We don't need to explicityl write the return types, but it helps make _i very clear.
	//
	fun abs(): Tensor = elementOperation({a -> Math.abs(a)})

	fun add(rhs: Float): Tensor = elementOperation({a -> a+rhs})

	fun add_i(rhs: Float): Unit = elementOperation_i({a -> a+rhs})

	fun add(rhs: Tensor): Tensor = elementOperation(rhs, {a,b -> a+b})

	fun add_i(rhs: Tensor): Unit = elementOperation_i(rhs, {a,b -> a+b})

	fun mul(c: Float): Tensor = elementOperation({a -> a*c})

	fun mul(rhs: Tensor): Tensor = elementOperation(rhs, {a, b -> a*b})

	fun neg(): Tensor = elementOperation({a -> -a})

	fun neg_i(): Unit = elementOperation_i({a -> -a})

	fun pow(c: Float): Tensor = elementOperation({a -> Math.pow(a.toDouble(), c.toDouble()).toFloat()})

	fun sign(): Tensor = elementOperation({ a -> Math.signum(a) })

	fun sub(rhs: Tensor): Tensor = elementOperation(rhs, {a, b -> a-b})

	fun sub_i(rhs: Tensor): Unit = elementOperation_i(rhs, {a, b -> a-b})

	fun tanh(): Tensor = elementOperation({a -> Math.tanh(a.toDouble()).toFloat()})

	fun tanh_i(): Unit = elementOperation_i({a -> Math.tanh(a.toDouble()).toFloat()})
}

fun Tensor.Companion.fromString(s:String): Tensor {
	var shapeString = s.substringAfter("TENSOR[").substringBefore("] : [")
	var dataString = s.substringAfter("] : [").substringBeforeLast("]")

	var t = Tensor(shapeString.split(", ").map { x -> x.toInt() }.toIntArray(), dataString.split(", ").map { x -> x.toFloat() }.toFloatArray())

	return t
}

fun Tensor.Companion.newFromFun(vararg shape: Int, initFunction: (Int)->Float): Tensor {
	return Tensor(shape, FloatArray(init = {index -> initFunction(index)}, size = shape.reduce { a, b -> a*b }))
}

fun Tensor.Companion.eye(vararg shape: Int): Tensor {
	return Tensor.newFromFun(*shape, initFunction = { a -> if(a % shape.last() == a / shape.last()) { 1.0f } else { 0.0f } })
}

fun Tensor.Companion.zeros(vararg shape: Int): Tensor {
	return Tensor(shape, FloatArray(size=shape.reduce { a, b -> a*b }))
}

fun Tensor.Companion.ones(vararg shape: Int): Tensor {
	return Tensor(shape, FloatArray(size=shape.reduce { a, b -> a*b }, init = { a->1.0f }))
}

fun Tensor.Companion.random(vararg shape: Int): Tensor {
	return Tensor.newFromFun(*shape, initFunction = { a -> Tensor.random.nextGaussian().toFloat() })
}

fun Tensor.Companion.indexToCoordinate(shape:IntArray, index:Int): IntArray {
	val dimensionOffset = IntArray(size=shape.size)
	for(index in (0..shape.size-2)) {
		dimensionOffset[index] = shape.slice(index+1..shape.size-1).reduce { a, b -> a*b }
	}
	dimensionOffset[shape.size-1] = 1

	// Treat the index as a 'count' item and iterate over the indices in order.
	var quantLeft = index
	val indexArray: MutableList<Int> = mutableListOf()
	for(dimensionSize in dimensionOffset) {
		indexArray.add(quantLeft / dimensionSize)
		quantLeft %= dimensionSize
	}

	return indexArray.toIntArray()
}
