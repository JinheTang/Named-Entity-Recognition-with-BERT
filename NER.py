from transformers import BertModel, BertPreTrainedModel
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.nn import CrossEntropyLoss
class BertNER(BertPreTrainedModel):
	def __init__(self, config):
		super(BertNER, self).__init__(config)
		self.num_labels = config.num_labels

		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, config.num_labels)

		self.init_weights()

	def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
				position_ids=None, inputs_embeds=None, head_mask=None):
		input_ids, input_token_starts = input_data
		outputs = self.bert(input_ids,
							attention_mask=attention_mask,
							token_type_ids=token_type_ids,
							position_ids=position_ids,
							head_mask=head_mask,
							inputs_embeds=inputs_embeds)
		sequence_output = outputs[0]

		origin_sequence_output = [
			layer[starts.nonzero().squeeze(1)]
			for layer, starts in zip(sequence_output, input_token_starts)]
		padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
		padded_sequence_output = self.dropout(padded_sequence_output)

		logits = self.classifier(padded_sequence_output)

		outputs = (logits,)
		if labels is not None:
			loss_mask = labels.gt(-1)
			loss_fct = CrossEntropyLoss()

			if loss_mask is not None:
				active_loss = loss_mask.view(-1) == 1
				active_logits = logits.view(-1, self.num_labels)[active_loss]
				active_labels = labels.view(-1)[active_loss]
				loss = loss_fct(active_logits, active_labels)
			else:
				loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			outputs = (loss,) + outputs

		return outputs
