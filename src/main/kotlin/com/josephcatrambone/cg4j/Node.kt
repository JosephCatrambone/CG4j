package com.josephcatrambone.cg4j

/**
 * Created by jcatrambone on 12/22/16.
 */
abstract class Node(val shape:IntArray, val inputs:Array<Node>) {
	var name:String = ""
	var id=-1
	abstract fun forwardOperation(vararg inputValues: Tensor): Tensor
	abstract fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor>
}

class InputNode(vararg shape:Int) : Node(shape, inputs=arrayOf<Node>()) {
	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		throw RuntimeException("You should never see this.  The graph should be checking the inputs before this gets called.")
	}

	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
		return arrayOf()
	}
}

class VariableNode(vararg shape:Int) : Node(shape, inputs=arrayOf<Node>()) {
	var value: Tensor = Tensor.zeros(*shape)

	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return value
	}

	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
		return arrayOf()
	}
}

class AddConstantNode(lhs:Node, val c:Float) : Node(lhs.shape, arrayOf<Node>(lhs)) {
	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return inputValues[0].add(c)
	}

	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
		return arrayOf(adjoint)
	}
}

class AddNode(lhs:Node, rhs:Node) : Node(lhs.shape, arrayOf<Node>(lhs, rhs)) {
	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return inputValues[0].add(inputValues[1])
	}

	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
		return arrayOf(adjoint, adjoint)
	}
}

class SubtractNode(lhs:Node, rhs:Node) : Node(lhs.shape, arrayOf<Node>(lhs, rhs)) {
	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return inputValues[0].sub(inputValues[1])
	}

	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
		return arrayOf(adjoint, adjoint.neg())
	}
}

class ElementMultiplyNode(lhs:Node, rhs:Node) : Node(lhs.shape, arrayOf<Node>(lhs, rhs)) {
	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return inputValues[0].mul(inputValues[1])
	}

	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
		return arrayOf(forwardValues[1].mul(adjoint), forwardValues[0].mul(adjoint))
	}
}

class MatrixMultiplyNode : Node {
	// TODO: Figure out how to better use the tensor product here.
	constructor(lhs:Node, rhs:Node) : super(shape = intArrayOf(lhs.shape[0], rhs.shape[1]), inputs = arrayOf<Node>(lhs, rhs)) {

	}

	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return inputValues[0].mmul(inputValues[1])
	}

	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
		return arrayOf(
			// Left adjoint. If C=AB, adj(a) = adj(c)*bT
			adjoint.mmul(forwardValues[1].transpose()),
			// Right adjoint.  adj(b) = aT*adj(c)
			forwardValues[0].transpose().mmul(adjoint)
		)
	}
}

class TanhNode(n:Node) : Node(shape=n.shape, inputs=arrayOf<Node>(n)) {
	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return inputValues[0].tanh()
	}

	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
		return arrayOf(
				adjoint.mul(Tensor.ones(*adjoint.shape).sub(forwardValues[0].tanh().mul(forwardValues[0].tanh())))
		)
	}
}

class PowerNode(base:Node, val exponent: Float) : Node(shape=base.shape, inputs=arrayOf<Node>(base)) {
	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return inputValues[0].pow(exponent)
	}

	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
		return arrayOf(
			adjoint.mul(forwardValues[0].pow(exponent-1.0f).mul(exponent))
		)
	}
}

class AbsNode(n:Node) : Node(n.shape, arrayOf<Node>(n)) {
	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return inputValues[0].abs()
	}

	override fun adjointOperation(forwardValues: Array<Tensor>, adjoint: Tensor): Array<Tensor> {
		return arrayOf(
			adjoint.mul(forwardValues[0].sign())
		)
	}
}

// And now add the actual node types.

// NOTE: THESE MAY BE MADE OBSOLETE BY THE BROADCAST OPERATOR
fun constantAdd(left: Int, cons: Float): Int {
	return addNode(
			"constant",
			shapes[left],
			intArrayOf(left),
			{ x -> x[0].add(cons) },
			{ forwards, parentAdjoint -> arrayOf(parentAdjoint) }
	)
}

fun constantMultiply(left: Int, cons: Float): Int {
	return addNode(
			"multiplyConstant",
			shapes[left],
			intArrayOf(left),
			{ x -> x[0].mul(cons) },
			{ forwards, parentAdjoint -> arrayOf(parentAdjoint.mul(cons)) }
	)
}
// END

fun addWithBroadcast(targetNodeToMatch: Int, operandToBroadcast: Int): Int {
	return addNode(
			"addWithBroadcast",
			shapes[targetNodeToMatch].copyOf(),
			intArrayOf(targetNodeToMatch, operandToBroadcast),
			{ throw NotImplementedError() },
			{ forwards, thisAdjoint ->
				throw NotImplementedError("Still working on backprop with broadcast.")
				arrayOf(
						// TODO: Start here.
				)
			}
	)
}

fun convolution2D(inputMatrix: Int, convolutionKernel: Int, stride: Int): Int {
	// Input: Volume of W1 H1 D1
	// Params: K -> # filters, F -> Spatial extent, S -> Stride, P -> Padding.
	// Output: W2 = (W1 - F + 2P)/S + 1 || H2 is like W2 || D2 = K
	throw NotImplementedError()
}

fun deconvolution2D(inputMatrix: Int, deconvolutionKernel: Int, stride: Int): Int {
	throw NotImplementedError()
}