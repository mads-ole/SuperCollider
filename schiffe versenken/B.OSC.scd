//=== SETUP OSC CONNECTION TO PYTHON SCRIPT ===\\

// ---------------------------------------------------------------- *
//Debug if SuperCollider isn't receiving OSC messages

//trace OSC messages being send
OSCFunc.trace(hideStatusMsg: true);

//retrieve IP adresse and port supercollider is listening on
NetAddr.localAddr //--> should be 127.0.0.1, 57120


// ---------------------------------------------------------------- *
//RECEIVE OSC MESSAGES FROM PYTHON AND CONVERT IT TO REQUIRED FORMAT
(
//initialize 2Darray (OSC data will be transferred into those)
~array2D_1 = Array2D.new(5, 5);
~array2D_2 = Array2D.new(5, 5);
(
OSCdef(\test, {|msg, time, addr, recvPort|
	var matrix1, matrix2, rowCount = 0;

	// seperating data list from message
	// by removing flag at first index of incoming list
	msg.remove(msg[0]);
	// msg.postln;

	//seperating the playing field data sets
	matrix1 = msg.copyFromStart(24);
	matrix2 = msg.copyToEnd(25);
	// matrix1.postln;
	// matrix2.postln;

	//convert (1st) incoming array into a 5x5 2Darray
	matrix1.do({
		arg item, ind;

		ind = ind % 5;
		~array2D_1[rowCount, ind] = item;

		if (ind == 4,
			{rowCount = rowCount + 1}
		);
	});

	rowCount = 0;
	//convert (2nd) incoming array into a 5x5 2Darray
	matrix2.do({
		arg item, ind;

		ind = ind % 5;
		~array2D_2[rowCount, ind] = item;

		if (ind == 4,
			{rowCount = rowCount + 1}
		);
	});


	~array2D_1.postln;
	~array2D_2.postln;
	"_".postln;

}, \test1);
)
)
